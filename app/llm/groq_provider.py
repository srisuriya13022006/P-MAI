import logging
import os
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on optional cloud dependency
    OpenAI = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqProvider:
    """Cloud model provider using Groq's OpenAI-compatible API with multi-model failover."""

    provider_name: str = "groq"
    provider_type: str = "cloud"

    def __init__(
        self,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        api_key: str | None = None,
    ):
        self.primary_model = model or settings.cloud_model
        if fallback_models is not None:
            self.fallback_models = list(fallback_models)
        else:
            self.fallback_models = list(getattr(settings, "cloud_fallback_models", []))
        self.model = self.primary_model
        self.last_used_model = self.primary_model
        self.model_cooldowns: dict[str, float] = {}

        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for Groq cloud inference. "
                "Install it with: pip install openai"
            )

        resolved_key = api_key or settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY is not set in the project environment.")

        self.client = OpenAI(
            api_key=resolved_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
            timeout=10.0,
        )
        self.last_usage: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self.last_status: str = "200_OK"
        self.last_retry_after: float | None = None

    def get_candidate_models(self) -> list[str]:
        """Return available models in priority order, bypassing models with active cooldowns."""
        now = time.monotonic()
        all_models = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        available = [m for m in all_models if now >= self.model_cooldowns.get(m, 0.0)]
        if not available:
            # If all models are currently in cooldown, attempt the one whose cooldown expires earliest
            available = sorted(all_models, key=lambda m: self.model_cooldowns.get(m, 0.0))
        return available

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)

        # Sensible max_tokens default (600 tokens) prevents Groq from reserving
        # 4,096 tokens per casual request against the 8,000 TPM limit
        effective_max_tokens = max_tokens if max_tokens is not None else 600

        candidate_models = self.get_candidate_models()
        last_exception: Exception | None = None

        for idx, candidate_model in enumerate(candidate_models):
            # Support mock clients in test suites (e.g., FakeClient with responses)
            if hasattr(self.client, "responses") and not hasattr(self.client, "chat"):
                response = self.client.responses.create(
                    model=candidate_model,
                    input=request_messages,
                )
                self.model = candidate_model
                self.last_used_model = candidate_model
                self.last_status = "200_OK"
                self.last_retry_after = None
                return getattr(response, "output_text", "")

            try:
                response = self.client.chat.completions.create(
                    model=candidate_model,
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
                self.model = candidate_model
                self.last_used_model = candidate_model
                # Clear cooldown on success
                self.model_cooldowns.pop(candidate_model, None)
                return response.choices[0].message.content or ""

            except Exception as e:
                last_exception = e
                self.last_usage = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                cat = classify_cloud_error(e)

                if cat == CloudErrorCategory.RATE_LIMIT:
                    self.last_status = "429_RATE_LIMIT"
                    retry_val = None
                    resp = getattr(e, "response", None)
                    if resp is not None and hasattr(resp, "headers"):
                        h = resp.headers
                        retry_val = h.get("retry-after") or h.get("x-ratelimit-reset-tokens")
                    if not retry_val:
                        import re
                        m = re.search(r"try again in ([\d\.]+)s", str(e))
                        if m:
                            retry_val = m.group(1)
                    try:
                        self.last_retry_after = float(retry_val) if retry_val is not None else None
                    except (ValueError, TypeError):
                        self.last_retry_after = None

                    cooldown_secs = self.last_retry_after if self.last_retry_after is not None else 60.0
                    self.model_cooldowns[candidate_model] = time.monotonic() + cooldown_secs

                    if idx < len(candidate_models) - 1:
                        next_model = candidate_models[idx + 1]
                        logger.warning(
                            "Groq model '%s' rate limited (429, retry_after=%.1fs). Switching to failover model '%s'.",
                            candidate_model,
                            cooldown_secs,
                            next_model,
                        )
                        continue
                else:
                    self.last_status = f"ERROR_{type(e).__name__}"
                    self.last_retry_after = None
                    if idx < len(candidate_models) - 1:
                        next_model = candidate_models[idx + 1]
                        logger.warning(
                            "Groq model '%s' encountered error (%s). Switching to failover model '%s'.",
                            candidate_model,
                            type(e).__name__,
                            next_model,
                        )
                        continue

        # If all candidates failed and client has responses fallback
        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=request_messages,
            )
            return getattr(response, "output_text", "")

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No available Groq models.")
