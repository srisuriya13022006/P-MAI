"""Memory extraction orchestrator for MAI.

Extraction strategy (tried in order):
  1. LLMMemoryExtractor  — semantic, handles natural language
  2. RegexMemoryExtractor — deterministic fallback for simple phrases

The public ``MemoryExtractor`` class coordinates both.  Either extractor
can be used in isolation (e.g. in tests) without the other.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.schemas.memory import MemoryCandidate

if TYPE_CHECKING:
    from app.memory.llm_extractor import LLMMemoryExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preserved rule-based extractor (unchanged logic from original implementation)
# ---------------------------------------------------------------------------


class RegexMemoryExtractor:
    """Deterministic, regex-based extractor — the original implementation.

    Used as a fallback when LLM extraction is unavailable or fails.
    """

    _patterns = (
        (r"\bmy name is ([A-Za-z][A-Za-z -]{0,60})[.!]?", "fact", "User's name is {}", 3),
        (r"\bi prefer ([^.!?]{1,100})", "preference", "User prefers {}", 3),
        (r"\bi like ([^.!?]{1,100})", "preference", "User likes {}", 2),
        (r"\bi use ([^.!?]{1,100})", "fact", "User uses {}", 2),
        (r"\bi live in ([A-Za-z][A-Za-z -]{0,60})[.!]?", "fact", "User lives in {}", 2),
        (r"\bi am learning ([^.!?]{1,100})", "goal", "User is learning {}", 2),
        (r"\bmy goal is to ([^.!?]{1,100})", "goal", "User's goal is to {}", 3),
        (r"\bi am building ([^.!?]{1,100})", "project", "User is building {}", 3),
    )

    def extract(self, message: str) -> list[MemoryCandidate]:
        candidates = []
        for pattern, memory_type, template, importance in self._patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                value = " ".join(match.group(1).split()).strip()
                value = re.sub(r"\s+now$", "", value, flags=re.IGNORECASE).strip()
                candidates.append(
                    MemoryCandidate(
                        content=template.format(value),
                        memory_type=memory_type,
                        importance=importance,
                    )
                )
        return candidates


# ---------------------------------------------------------------------------
# Hybrid orchestrator — public API (backward compatible)
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """Hybrid memory extractor: LLM-first with regex fallback.

    If *llm_extractor* is supplied and produces at least one candidate,
    those results are returned.  If it produces nothing or fails, the
    *RegexMemoryExtractor* is used as the fallback.

    When *llm_extractor* is ``None`` the class behaves identically to
    the old ``MemoryExtractor`` — only regex extraction is performed.
    This preserves full backward compatibility.
    """

    def __init__(self, llm_extractor: LLMMemoryExtractor | None = None) -> None:
        self._llm = llm_extractor
        self._regex = RegexMemoryExtractor()

    def extract(self, message: str) -> list[MemoryCandidate]:
        if self._llm is not None:
            try:
                candidates = self._llm.extract(message)
                if candidates:
                    logger.debug(
                        "LLM extractor produced %d candidate(s) — skipping regex fallback.",
                        len(candidates),
                    )
                    return candidates
                logger.debug(
                    "LLM extractor returned no candidates — using regex fallback."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM extractor raised unexpectedly (%s) — using regex fallback.", exc
                )

        candidates = self._regex.extract(message)
        if candidates:
            logger.debug("Regex fallback extractor produced %d candidate(s).", len(candidates))
        return candidates