"""LLM-based memory extraction for MAI.

Architecture
------------
LLMMemoryExtractor
    ↓  calls
LLMExtractionProvider (protocol / abstraction)
    ↓  implemented by
_OllamaExtractionProvider  (default — wraps the existing OllamaProvider contract)

The extractor is completely decoupled from the provider implementation.
Swap models by passing a different LLMExtractionProvider at construction time.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, ValidationError

from app.prompts.memory import MEMORY_EXTRACTION_SYSTEM_PROMPT
from app.schemas.memory import MemoryCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

SUPPORTED_MEMORY_TYPES = frozenset({"fact", "preference", "goal", "project"})


@runtime_checkable
class LLMExtractionProvider(Protocol):
    """Minimal interface required by LLMMemoryExtractor.

    Any object that has a ``generate(messages, system_prompt)`` method
    satisfies this protocol — including OllamaProvider and GroqProvider.
    """

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Internal validated schema for LLM output
# ---------------------------------------------------------------------------


class _RawMemoryItem(BaseModel):
    type: str
    content: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in SUPPORTED_MEMORY_TYPES:
            raise ValueError(
                f"Unsupported memory type '{value}'. "
                f"Must be one of: {sorted(SUPPORTED_MEMORY_TYPES)}"
            )
        return normalised

    @field_validator("content")
    @classmethod
    def _clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Memory content must not be blank.")
        return cleaned


class _LLMExtractionResult(BaseModel):
    """Wrapper that stores the raw list; per-item validation is done lazily."""

    memories: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM extractor
# ---------------------------------------------------------------------------


class LLMMemoryExtractor:
    """Extracts persistent user memories using an LLM provider.

    Falls back gracefully — if extraction fails for any reason (timeout,
    invalid JSON, schema violation) it returns an empty list so the caller
    can delegate to the rule-based extractor.
    """

    def __init__(self, llm: LLMExtractionProvider) -> None:
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, message: str) -> list[MemoryCandidate]:
        """Return a list of MemoryCandidate objects extracted from *message*.

        Returns an empty list on any failure — never raises.
        """
        logger.debug("LLM memory extraction started for message (len=%d)", len(message))

        raw_output = self._call_llm(message)
        if not raw_output:
            logger.debug("LLM returned empty response — no memories extracted.")
            return []

        parsed = self._parse_json(raw_output)
        if parsed is None:
            logger.debug("LLM output could not be parsed as JSON.")
            return []

        validated = self._validate(parsed)
        if validated is None:
            logger.debug("LLM extraction result failed schema validation.")
            return []

        candidates = self._to_candidates(validated)
        logger.debug("LLM extraction completed — %d valid candidate(s) found.", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, message: str) -> str:
        """Call the LLM provider. Returns empty string on any failure."""
        try:
            return self.llm.generate(
                messages=[{"role": "user", "content": message}],
                system_prompt=MEMORY_EXTRACTION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM memory extraction call failed: %s", exc)
            return ""

    def _parse_json(self, text: str) -> dict | None:
        """Extract the first JSON object from *text* and parse it.

        Handles model responses that wrap JSON in markdown fences.
        """
        # Strip common markdown code fences the model might add
        stripped = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()

        # Find the first {...} block
        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not brace_match:
            logger.debug("No JSON object found in LLM output: %r", text[:200])
            return None

        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError as exc:
            logger.debug("JSON decode error in LLM output: %s | text=%r", exc, text[:200])
            return None

    def _validate(self, data: dict) -> _LLMExtractionResult | None:
        """Validate parsed dict against the extraction schema."""
        try:
            return _LLMExtractionResult.model_validate(data)
        except ValidationError as exc:
            logger.debug("LLM extraction schema validation failed: %s", exc)
            return None

    def _to_candidates(self, result: _LLMExtractionResult) -> list[MemoryCandidate]:
        """Convert validated raw items to MemoryCandidate objects, skipping bad ones."""
        candidates: list[MemoryCandidate] = []
        for raw in result.memories:
            # Validate each item individually so one bad item doesn't poison the batch.
            try:
                item = _RawMemoryItem.model_validate(raw)
            except (ValidationError, TypeError) as exc:
                logger.debug("Skipping invalid memory item from LLM: %s", exc)
                continue
            try:
                candidates.append(
                    MemoryCandidate(
                        content=item.content,
                        memory_type=item.type,  # type: ignore[arg-type]
                        importance=self._importance_from_confidence(item.confidence),
                        confidence=item.confidence,
                    )
                )
            except (ValidationError, ValueError) as exc:
                logger.debug("Skipping invalid memory candidate: %s", exc)
        return candidates

    @staticmethod
    def _importance_from_confidence(confidence: float) -> int:
        """Map confidence 0–1 to importance 1–3."""
        if confidence >= 0.95:
            return 3
        if confidence >= 0.80:
            return 2
        return 1
