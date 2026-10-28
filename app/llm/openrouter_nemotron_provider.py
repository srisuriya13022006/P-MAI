"""
P16.1 — OpenRouter NVIDIA Nemotron Provider Adapter.
Dedicated LLM provider adapter for procedural-memory and reusable-skill operations.
Does not alter global primary chat routing (Gemini -> Groq -> Ollama).
"""
import logging
import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenRouterNemotronProvider:
    """
    Dedicated LLM provider adapter for OpenRouter's NVIDIA Nemotron model.
    Used exclusively for procedural memory, structured skill extraction, and plan adaptation.
    """

    provider_name: str = "openrouter"
    provider_type: str = "cloud"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        client: Any | None = None,
    ):
        self.model = model or getattr(settings, "openrouter_nemotron_model", "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.base_url = base_url or getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
        self.timeout = timeout if timeout is not None else getattr(settings, "openrouter_timeout_seconds", 30.0)
        self.default_max_tokens = max_tokens if max_tokens is not None else getattr(settings, "openrouter_max_tokens", 1500)
        self.last_used_model = self.model

        self.last_usage: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self.last_status: str = "200_OK"
        self.last_retry_after: float | None = None

        if client is not None:
            self.client = client
        else:
            if OpenAI is None:
                raise ImportError(
                    "The 'openai' package is required for OpenRouter inference. "
                    "Install it with: pip install openai"
                )

            resolved_key = (
                api_key
                or (getattr(settings, "get_openrouter_api_key", None)() if hasattr(settings, "get_openrouter_api_key") else None)
                or getattr(settings, "openrouter_api_key", None)
                or getattr(settings, "open_router_api", None)
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPEN_ROUTER_API")
            )
            if not resolved_key:
                raise ValueError("OPENROUTER_API_KEY is not set in the project environment.")

            self.client = OpenAI(
                api_key=resolved_key,
                base_url=self.base_url,
                max_retries=0,
                timeout=self.timeout,
                default_headers={
                    "HTTP-Referer": "https://p-mai.local",
                    "X-Title": "P-MAI Procedural Memory",
                },
            )

    def _strip_reasoning_blocks(self, text: str) -> str:
        """Strip internal reasoning tags (<thought> or <think>) if present."""
        if not text:
            return ""
        # Remove <thought>...</thought> and <think>...</think>
        cleaned = re.sub(r"<(?:thought|think)>.*?</(?:thought|think)>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate completion using OpenRouter Nemotron.
        Accepts standard role-based messages and returns stripped assistant text.
        """
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)

        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        # Support mock clients in tests
        if hasattr(self.client, "responses") and not hasattr(self.client, "chat"):
            response = self.client.responses.create(
                model=self.model,
                input=request_messages,
            )
            self.last_status = "200_OK"
            self.last_retry_after = None
            raw_text = getattr(response, "output_text", "")
            return self._strip_reasoning_blocks(raw_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                max_tokens=effective_max_tokens,
            )

            usage = getattr(response, "usage", None)
            if usage:
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            else:
                self.last_usage = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }

            self.last_status = "200_OK"
            self.last_retry_after = None

            if not response.choices:
                return ""

            msg = response.choices[0].message
            raw_content = getattr(msg, "content", "") or ""
            return self._strip_reasoning_blocks(raw_content)

        except Exception as e:
            self.last_usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            err_str = str(e).lower()

            if "401" in err_str or "unauthorized" in err_str or "invalid api key" in err_str:
                self.last_status = "401_UNAUTHORIZED"
                logger.error("OpenRouter Nemotron authentication failure: check OPENROUTER_API_KEY.")
            elif "429" in err_str or "rate limit" in err_str:
                self.last_status = "429_RATE_LIMIT"
                logger.warning("OpenRouter Nemotron rate limited (429).")
            elif "timeout" in err_str:
                self.last_status = "408_TIMEOUT"
                logger.warning("OpenRouter Nemotron request timed out.")
            else:
                self.last_status = f"ERROR_{type(e).__name__}"
                logger.error("OpenRouter Nemotron request failed: %s", type(e).__name__)

            raise
