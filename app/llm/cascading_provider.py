"""
P15.5 — Cascading Cloud LLM Provider.
Chains multiple cloud LLM providers (e.g. Gemini primary -> Groq failover).
If the primary cloud model raises any error (rate limit, quota, server error, timeout),
it seamlessly switches to the next configured cloud provider before resorting to local fallback.
"""
import logging
import time
from typing import Any

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class CascadingCloudProvider:
    """
    Manages an ordered failover sequence across multiple cloud LLM providers.
    Example: Gemini (primary) -> Groq (failover).
    """

    provider_type: str = "cloud"

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("CascadingCloudProvider requires at least one LLMProvider.")
        self.providers = list(providers)
        self.active_provider: LLMProvider = self.providers[0]
        self.provider_cooldowns: dict[str, float] = {}
        self.last_failover_reason: str | None = None
        self.last_failover_occurred: bool = False

    @property
    def provider_name(self) -> str:
        return self.active_provider.provider_name

    @property
    def model(self) -> str:
        return getattr(self.active_provider, "model", "unknown")

    @property
    def last_usage(self) -> dict[str, int | None]:
        return getattr(
            self.active_provider,
            "last_usage",
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        )

    @property
    def last_status(self) -> str:
        return getattr(self.active_provider, "last_status", "200_OK")

    @property
    def last_retry_after(self) -> float | None:
        return getattr(self.active_provider, "last_retry_after", None)

    def get_candidate_providers(self) -> list[LLMProvider]:
        """Return providers in priority order, bypassing providers with active cooldowns if others are ready."""
        now = time.monotonic()
        ready = [p for p in self.providers if now >= self.provider_cooldowns.get(p.provider_name, 0.0)]
        if not ready:
            # If all are cooling down, sort by earliest cooldown expiration
            return sorted(self.providers, key=lambda p: self.provider_cooldowns.get(p.provider_name, 0.0))
        # Keep original priority order among ready providers
        return ready

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        candidate_providers = self.get_candidate_providers()
        last_exception: Exception | None = None
        self.last_failover_occurred = False
        self.last_failover_reason = None

        for idx, provider in enumerate(candidate_providers):
            try:
                try:
                    res = provider.generate(
                        messages,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except TypeError:
                    res = provider.generate(
                        messages,
                        system_prompt=system_prompt,
                    )

                self.active_provider = provider
                # Clear cooldown on success
                self.provider_cooldowns.pop(provider.provider_name, None)
                return res

            except Exception as err:
                last_exception = err
                error_name = type(err).__name__
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                cat = classify_cloud_error(err)

                cooldown = 30.0
                retry_after = getattr(provider, "last_retry_after", None)
                if retry_after is not None and retry_after > 0:
                    cooldown = retry_after
                elif cat == CloudErrorCategory.RATE_LIMIT:
                    cooldown = 60.0

                self.provider_cooldowns[provider.provider_name] = time.monotonic() + cooldown

                if idx < len(candidate_providers) - 1:
                    next_provider = candidate_providers[idx + 1]
                    self.last_failover_occurred = True
                    self.last_failover_reason = f"{provider.provider_name} ({error_name}): {err}"
                    logger.warning(
                        "Cloud provider '%s' (model '%s') failed (%s: %s). Switching to failover cloud provider '%s' (model '%s').",
                        provider.provider_name,
                        getattr(provider, "model", "unknown"),
                        error_name,
                        err,
                        next_provider.provider_name,
                        getattr(next_provider, "model", "unknown"),
                    )
                    continue

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No available cloud providers.")
