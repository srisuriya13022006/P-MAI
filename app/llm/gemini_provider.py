"""
P15.5 / P15.7 — Google Gemini Cloud LLM Provider with Quota-Aware Multi-Model Pool.
Provides robust inference using Google's OpenAI-compatible endpoint with:
1. Quota-aware model pool (gemini-3.5-flash-lite, gemini-3.1-flash-lite, etc.).
2. Deterministic fair load balancing across healthy candidates.
3. Fine-grained classification of RPM/TPM vs. RPD daily exhaustion.
4. Fast bypass of exhausted models until Pacific Time midnight reset.
5. Immediate project-wide restriction detection.
6. Zero API key leakage.
"""
import logging
import os
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from app.core.config import settings
from app.llm.gemini_pool import (
    GeminiErrorType,
    GeminiModelHealth,
    GeminiModelPool,
    GeminiModelStatus,
    classify_gemini_error,
)

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Cloud model provider using Google's Gemini OpenAI-compatible API with quota-aware multi-model pool."""

    provider_name: str = "gemini"
    provider_type: str = "cloud"

    def __init__(
        self,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        model_pool: list[str] | None = None,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    ):
        configured_pool = model_pool or getattr(settings, "gemini_model_pool", None)
        self.primary_model = model or getattr(settings, "gemini_model", "gemini-3.5-flash-lite")
        if fallback_models is not None:
            self.fallback_models = list(fallback_models)
        else:
            self.fallback_models = list(getattr(settings, "gemini_fallback_models", ["gemini-3.1-flash-lite"]))

        self.pool = GeminiModelPool(
            models=configured_pool if model is None and fallback_models is None else None,
            primary_model=self.primary_model,
            fallback_models=self.fallback_models,
        )

        self.model = self.primary_model
        self.last_used_model = self.primary_model
        self.base_url = base_url

        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for Gemini cloud inference. "
                "Install it with: pip install openai"
            )

        resolved_key = (
            api_key
            or getattr(settings, "gemini_api_key", None)
            or getattr(settings, "google_api_key", None)
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in the project environment.")

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=15.0,
        )
        self.last_usage: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self.last_status: str = "200_OK"
        self.last_retry_after: float | None = None

    @property
    def model_cooldowns(self) -> dict[str, float]:
        """Backward-compatibility mapping of model cooldown timestamps."""
        return {
            m: h.cooldown_until
            for m, h in self.pool.models_health.items()
            if h.cooldown_until > 0.0
        }

    @property
    def pool_status(self) -> dict[str, Any]:
        """Return safe status of all models in the pool."""
        return self.pool.get_pool_status()

    def get_candidate_models(self) -> list[str]:
        """Return healthy candidate models ordered by LRU for fair quota distribution."""
        return self.pool.get_candidate_models()

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)

        effective_max_tokens = max_tokens if max_tokens is not None else 800

        # 1. Check if project-wide limit is active
        if self.pool.is_project_limited():
            logger.warning("Gemini project-wide quota limit active. Bypassing Gemini pool.")
            raise RuntimeError("Gemini project-wide quota or billing restriction active.")

        candidate_models = self.get_candidate_models()
        last_exception: Exception | None = None

        for idx, candidate_model in enumerate(candidate_models):
            # Support mock clients in test suites
            if hasattr(self.client, "responses") and not hasattr(self.client, "chat"):
                t0 = time.perf_counter()
                response = self.client.responses.create(
                    model=candidate_model,
                    input=request_messages,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.model = candidate_model
                self.last_used_model = candidate_model
                self.last_status = "200_OK"
                self.last_retry_after = None
                self.pool.record_success(candidate_model, elapsed)
                return getattr(response, "output_text", "")

            try:
                t0 = time.perf_counter()
                create_kwargs: dict[str, Any] = {
                    "model": candidate_model,
                    "messages": request_messages,
                    "max_tokens": effective_max_tokens,
                }
                if "tools" in kwargs:
                    create_kwargs["tools"] = kwargs["tools"]
                if "tool_choice" in kwargs:
                    create_kwargs["tool_choice"] = kwargs["tool_choice"]

                response = self.client.chat.completions.create(**create_kwargs)
                elapsed = (time.perf_counter() - t0) * 1000.0

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
                self.model = candidate_model
                self.last_used_model = candidate_model
                self.pool.record_success(candidate_model, elapsed, self.last_usage.get("total_tokens"))

                choice = response.choices[0]
                if getattr(choice.message, "tool_calls", None) and not choice.message.content:
                    import json
                    calls = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in choice.message.tool_calls
                    ]
                    return json.dumps(calls)

                return choice.message.content or ""

            except Exception as e:
                last_exception = e
                self.last_usage = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                error_type, retry_after, reason = classify_gemini_error(e)
                self.pool.record_failure(candidate_model, error_type, retry_after=retry_after, reason=reason)
                self.last_status = error_type.value
                self.last_retry_after = retry_after

                # Hard configuration / authentication errors must fail immediately
                if error_type == GeminiErrorType.AUTH_ERROR:
                    logger.error("Gemini authentication failure on model '%s': %s", candidate_model, e)
                    raise e

                # Project-wide limit must immediately trigger downstream fallback without cycling models
                if error_type == GeminiErrorType.PROJECT_WIDE_LIMIT:
                    logger.error("Gemini project-wide limit: skipping remaining Gemini models.")
                    raise e

                # For model-specific RPM, RPD, or 404, log and continue to next candidate in pool
                if idx < len(candidate_models) - 1:
                    next_model = candidate_models[idx + 1]
                    logger.warning(
                        "Gemini model '%s' failed (%s: %s). Switching to next Gemini candidate '%s'.",
                        candidate_model,
                        error_type.value,
                        reason,
                        next_model,
                    )
                    continue

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No available Gemini models in pool.")
