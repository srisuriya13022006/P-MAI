"""
P15.4 — Resilient Cloud-First LLM Provider with Controlled Local Fallback.
Provides bounded primary-to-fallback resolution without provider flapping loops or secret leakage.
"""
import logging
import time
from typing import Any

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ResilientLLMProvider:
    """
    Coordinates primary Cloud LLM execution with controlled Local Qwen/Ollama fallback.
    Prevents provider flapping loops within requests and exposes clean runtime mode observability.
    """

    def __init__(
        self,
        primary_provider: LLMProvider | None,
        fallback_provider: LLMProvider | None,
        fallback_enabled: bool = True,
    ):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.fallback_enabled = fallback_enabled

        self.runtime_mode: str = "CLOUD" if primary_provider is not None else "LOCAL_FALLBACK"
        self.last_provider_used: str = primary_provider.provider_name if primary_provider else (
            fallback_provider.provider_name if fallback_provider else "none"
        )
        self.last_latency_ms: float = 0.0
        self.cloud_call_count: int = 0
        self.fallback_call_count: int = 0
        self.last_telemetry: dict[str, Any] = {}
        self.turn_records: list[dict[str, Any]] = []

    def reset_turn_accounting(self) -> None:
        """Clear call records for the start of a new conversational turn."""
        self.turn_records.clear()

    def get_turn_accounting(self) -> dict[str, Any]:
        """Return safe per-turn LLM call accounting telemetry."""
        groq_calls = [r for r in self.turn_records if r.get("provider") == "groq"]
        return {
            "groq_call_count": len(groq_calls),
            "total_calls": len(self.turn_records),
            "fallback_occurred": any(r.get("fallback_occurred", False) for r in self.turn_records),
            "total_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in self.turn_records),
            "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in self.turn_records),
            "total_tokens": sum(r.get("total_tokens") or 0 for r in self.turn_records),
            "calls": list(self.turn_records),
        }

    @property
    def provider_name(self) -> str:
        return self.last_provider_used

    @property
    def provider_type(self) -> str:
        if self.runtime_mode == "CLOUD":
            return "cloud"
        return "local"

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        stage: str = "conversation",
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate completion using Primary (Cloud) LLM first, falling back to Local (Ollama) if primary fails.
        Enforces single fallback transition and zero loops within a single request.
        """
        start = time.perf_counter()
        fallback_reason: str | None = None
        primary_status: str = "200_OK"

        # 1. Try Primary Cloud Provider if configured
        if self.primary_provider is not None:
            self.cloud_call_count += 1
            primary_start = time.perf_counter()
            try:
                try:
                    res = self.primary_provider.generate(messages, system_prompt=system_prompt, max_tokens=max_tokens)
                except TypeError:
                    res = self.primary_provider.generate(messages, system_prompt=system_prompt)
                
                total_ms = round((time.perf_counter() - start) * 1000.0, 2)
                self.last_latency_ms = total_ms
                self.runtime_mode = "CLOUD"
                self.last_provider_used = self.primary_provider.provider_name

                # Extract token usage from primary provider if available
                usage = getattr(self.primary_provider, "last_usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

                call_record = {
                    "stage": stage,
                    "provider": self.primary_provider.provider_name,
                    "model": getattr(self.primary_provider, "model", "unknown"),
                    "http_status": "200_OK",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "elapsed_ms": total_ms,
                    "fallback_occurred": False,
                    "fallback_reason": None,
                }
                self.turn_records.append(call_record)

                self.last_telemetry = {
                    "stage": stage,
                    "status": "200_OK",
                    "active_provider": self.primary_provider.provider_name,
                    "active_model": getattr(self.primary_provider, "model", "unknown"),
                    "cloud_call_count": self.cloud_call_count,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "fallback_reason": None,
                    "fallback_provider_latency_ms": None,
                    "total_elapsed_ms": total_ms,
                }
                logger.info(
                    "LLM Telemetry: stage=%s status=200_OK provider=%s model=%s prompt_tokens=%s completion_tokens=%s cloud_calls=%d total_ms=%.1f",
                    stage,
                    self.primary_provider.provider_name,
                    getattr(self.primary_provider, "model", "unknown"),
                    prompt_tokens,
                    completion_tokens,
                    self.cloud_call_count,
                    total_ms,
                )
                return res
            except Exception as primary_err:
                error_type = type(primary_err).__name__
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                category = classify_cloud_error(primary_err)
                retry_after = getattr(self.primary_provider, "last_retry_after", None)

                # If 429 and retry_after guidance is tiny (<= 0.5s), honor retry once
                if category == CloudErrorCategory.RATE_LIMIT and retry_after is not None and 0 < retry_after <= 0.5:
                    logger.info("Honoring small provider retry guidance: sleeping %.2fs before one retry", retry_after)
                    time.sleep(retry_after)
                    try:
                        self.cloud_call_count += 1
                        try:
                            res = self.primary_provider.generate(messages, system_prompt=system_prompt, max_tokens=max_tokens)
                        except TypeError:
                            res = self.primary_provider.generate(messages, system_prompt=system_prompt)
                        total_ms = round((time.perf_counter() - start) * 1000.0, 2)
                        self.last_latency_ms = total_ms
                        self.runtime_mode = "CLOUD"
                        self.last_provider_used = self.primary_provider.provider_name
                        usage = getattr(self.primary_provider, "last_usage", {}) or {}
                        call_record = {
                            "stage": stage,
                            "provider": self.primary_provider.provider_name,
                            "model": getattr(self.primary_provider, "model", "unknown"),
                            "http_status": "200_OK",
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                            "elapsed_ms": total_ms,
                            "fallback_occurred": False,
                            "fallback_reason": None,
                        }
                        self.turn_records.append(call_record)
                        return res
                    except Exception as retry_err:
                        primary_err = retry_err
                        category = classify_cloud_error(primary_err)

                primary_status = "429_RATE_LIMIT" if category == CloudErrorCategory.RATE_LIMIT else f"ERROR_{error_type}"

                # Hard configuration / Auth errors (401, 403, missing/invalid key) must NEVER silently fall back
                if category == CloudErrorCategory.HARD_CONFIG:
                    self.runtime_mode = "OFFLINE"
                    logger.error(
                        "Primary LLM %s failed with hard configuration/auth error (%s). Fallback inhibited.",
                        getattr(self.primary_provider, "provider_name", "cloud"),
                        error_type,
                    )
                    raise primary_err

                if not self.fallback_enabled or self.fallback_provider is None:
                    self.runtime_mode = "OFFLINE"
                    logger.error(
                        "Primary LLM %s failed (%s) and fallback is disabled.",
                        getattr(self.primary_provider, "provider_name", "cloud"),
                        error_type,
                    )
                    raise primary_err

                # Controlled one-way transition to fallback
                if category == CloudErrorCategory.RATE_LIMIT:
                    fallback_reason = f"rate_limit_429 (retry_after={retry_after}s)" if retry_after else "rate_limit_429"
                else:
                    fallback_reason = error_type

                call_record = {
                    "stage": stage,
                    "provider": getattr(self.primary_provider, "provider_name", "cloud"),
                    "model": getattr(self.primary_provider, "model", "unknown"),
                    "http_status": primary_status,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "elapsed_ms": round((time.perf_counter() - primary_start) * 1000.0, 2),
                    "fallback_occurred": True,
                    "fallback_reason": fallback_reason,
                }
                self.turn_records.append(call_record)

                logger.warning(
                    "LLM provider: %s\nLLM fallback reason: %s\nLLM provider switched to: %s",
                    getattr(self.primary_provider, "provider_name", "cloud"),
                    fallback_reason,
                    getattr(self.fallback_provider, "provider_name", "ollama"),
                )
                self.runtime_mode = "LOCAL_FALLBACK"

        # 2. Local Fallback Provider
        if self.fallback_provider is not None:
            self.fallback_call_count += 1
            try:
                fb_start = time.perf_counter()
                try:
                    res = self.fallback_provider.generate(messages, system_prompt=system_prompt, max_tokens=max_tokens)
                except TypeError:
                    res = self.fallback_provider.generate(messages, system_prompt=system_prompt)
                fb_ms = round((time.perf_counter() - fb_start) * 1000.0, 2)
                total_ms = round((time.perf_counter() - start) * 1000.0, 2)
                self.last_latency_ms = total_ms
                self.last_provider_used = self.fallback_provider.provider_name

                fb_record = {
                    "stage": stage,
                    "provider": self.fallback_provider.provider_name,
                    "model": getattr(self.fallback_provider, "model", "unknown"),
                    "http_status": "200_OK",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "elapsed_ms": fb_ms,
                    "fallback_occurred": True,
                    "fallback_reason": fallback_reason,
                }
                self.turn_records.append(fb_record)

                self.last_telemetry = {
                    "stage": stage,
                    "status": primary_status,
                    "active_provider": self.fallback_provider.provider_name,
                    "active_model": getattr(self.fallback_provider, "model", "unknown"),
                    "cloud_call_count": self.cloud_call_count,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "fallback_reason": fallback_reason,
                    "fallback_provider_latency_ms": fb_ms,
                    "total_elapsed_ms": total_ms,
                }
                logger.warning(
                    "LLM Telemetry: stage=%s status=%s primary=%s switched_to=%s fallback_reason=%s fb_latency_ms=%.1f total_ms=%.1f cloud_calls=%d",
                    stage,
                    primary_status,
                    getattr(self.primary_provider, "provider_name", "cloud"),
                    self.fallback_provider.provider_name,
                    fallback_reason,
                    fb_ms,
                    total_ms,
                    self.cloud_call_count,
                )
                return res
            except Exception as fb_err:
                self.runtime_mode = "OFFLINE"
                logger.error(
                    "Local fallback LLM %s also failed (%s). System is OFFLINE.",
                    getattr(self.fallback_provider, "provider_name", "local"),
                    type(fb_err).__name__,
                )
                raise

        self.runtime_mode = "OFFLINE"
        raise RuntimeError("No LLM provider available (both primary and fallback are unconfigured or failed).")
