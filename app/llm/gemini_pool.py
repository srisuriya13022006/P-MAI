"""
P15.7 — Quota-Aware Multi-Model Gemini Model Pool.
Manages a resilient, quota-aware pool of Google Gemini models with:
1. Deterministic load distribution (LRU / fair balancing across healthy candidates).
2. Distinction between model-specific RPM/TPM rate limits vs. RPD daily exhaustion.
3. Fast bypass of exhausted models until Pacific Time midnight reset.
4. Immediate detection of project-wide account/billing restrictions to prevent pointless rotation.
5. Zero credential leakage.
"""
from dataclasses import dataclass, field
import datetime
from enum import Enum
import logging
import re
import time
from typing import Any
import zoneinfo

logger = logging.getLogger(__name__)


class GeminiModelStatus(str, Enum):
    HEALTHY = "HEALTHY"
    COOLDOWN_RPM = "COOLDOWN_RPM"
    EXHAUSTED_RPD = "EXHAUSTED_RPD"
    PROJECT_LIMITED = "PROJECT_LIMITED"
    UNSUPPORTED = "UNSUPPORTED"
    AUTH_ERROR = "AUTH_ERROR"


class GeminiErrorType(str, Enum):
    RPM_TPM_LIMIT = "RPM_TPM_LIMIT"
    RPD_EXHAUSTED = "RPD_EXHAUSTED"
    PROJECT_WIDE_LIMIT = "PROJECT_WIDE_LIMIT"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    AUTH_ERROR = "AUTH_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    UNRECOVERABLE = "UNRECOVERABLE"


def get_pacific_midnight_timestamp() -> float:
    """
    Return epoch timestamp for next Midnight Pacific Time (00:00 PST/PDT).
    Google AI Studio free-tier quotas reset daily at midnight Pacific Time.
    """
    try:
        pt_zone = zoneinfo.ZoneInfo("America/Los_Angeles")
    except Exception:
        pt_zone = datetime.timezone(datetime.timedelta(hours=-8))
    now = datetime.datetime.now(pt_zone)
    tomorrow = now.date() + datetime.timedelta(days=1)
    reset_dt = datetime.datetime.combine(tomorrow, datetime.time.min, tzinfo=pt_zone)
    return reset_dt.timestamp()


def classify_gemini_error(exc: Exception) -> tuple[GeminiErrorType, float | None, str]:
    """
    Classify Gemini API errors with high specificity.
    Returns: (error_type, retry_after_seconds, diagnostic_reason)
    """
    err_str = str(exc)
    err_lower = err_str.lower()
    err_type = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)

    # 1. Authentication / Permission Failures (401, 403)
    if status_code in (401, 403) or any(
        k in err_lower for k in ("unauthorized", "invalid api key", "forbidden", "permission denied", "api_key_invalid")
    ):
        return GeminiErrorType.AUTH_ERROR, None, "Authentication or permission failure"

    # 2. Unsupported / Deprecated Models (404)
    if status_code == 404 or "not found" in err_lower or "is not found for api version" in err_lower:
        return GeminiErrorType.UNSUPPORTED_MODEL, None, "Model ID not supported on current API version/endpoint"

    # Extract retry guidance if present
    retry_secs: float | None = None
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        h = resp.headers
        val = h.get("retry-after") or h.get("x-ratelimit-reset-tokens")
        if val:
            try:
                retry_secs = float(val)
            except (ValueError, TypeError):
                pass

    if retry_secs is None:
        # Check error JSON details for retryDelay (e.g. "retryDelay": "63529s" or retryDelay: '63529s')
        m_delay = re.search(r"['\"]?retryDelay['\"]?\s*:\s*['\"]?(\d+)(?:\.\d+)?s['\"]?", err_str)
        if m_delay:
            try:
                retry_secs = float(m_delay.group(1))
            except (ValueError, TypeError):
                pass

    if retry_secs is None:
        # Check text string for hours/minutes/seconds
        m_hms = re.search(r"retry in (\d+)h(\d+)m", err_lower)
        if m_hms:
            hours, mins = int(m_hms.group(1)), int(m_hms.group(2))
            retry_secs = float(hours * 3600 + mins * 60)
        else:
            m_sec = re.search(r"try again in ([\d\.]+)s", err_lower)
            if m_sec:
                try:
                    retry_secs = float(m_sec.group(1))
                except (ValueError, TypeError):
                    pass

    # 3. Rate Limits & Quota Exhaustion (429 / RESOURCE_EXHAUSTED)
    if status_code == 429 or "resource_exhausted" in err_lower or "rate limit" in err_lower or "quota exceeded" in err_lower:
        # Check for Project-Wide limits (billing disabled, project quota exhausted across models)
        if (
            "billing" in err_lower
            or "account disabled" in err_lower
            or "generaterequestsperdayperproject-" in err_lower
            or ("project" in err_lower and "permodel" not in err_lower and "quota exceeded for metric" in err_lower)
        ):
            return GeminiErrorType.PROJECT_WIDE_LIMIT, retry_secs, "Project-wide quota or billing restriction"

        # Check for Model-Specific RPD (Daily) Exhaustion:
        # Indicators: "permodel", "per day", "free_tier_requests", or retryDelay > 300s
        if (
            "permodel" in err_lower
            or "perday" in err_lower
            or "per day" in err_lower
            or "generate_content_free_tier_requests" in err_lower
            or (retry_secs is not None and retry_secs > 300.0)
        ):
            return GeminiErrorType.RPD_EXHAUSTED, retry_secs, "Model-specific daily request quota (RPD) exhausted"

        # Model-Specific RPM / TPM Transient Limit
        return GeminiErrorType.RPM_TPM_LIMIT, retry_secs, "Model-specific RPM/TPM rate limit"

    # 4. Server Errors & Network Timeouts
    if any(k in err_type for k in ("timeout", "connection", "connect", "unavailable")):
        return GeminiErrorType.SERVER_ERROR, 5.0, "Network timeout or connection drop"
    if status_code in (500, 502, 503, 504) or any(k in err_lower for k in ("internal server error", "503", "502", "504")):
        return GeminiErrorType.SERVER_ERROR, 5.0, "Transient Google Gemini server error"

    return GeminiErrorType.UNRECOVERABLE, None, f"Unclassified error: {type(exc).__name__}"


@dataclass
class GeminiModelHealth:
    """Per-model availability state, metrics, and cooldown tracking."""
    model_id: str
    status: GeminiModelStatus = GeminiModelStatus.HEALTHY
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    last_status_code: str = "200_OK"
    last_failure_reason: str | None = None
    last_failure_timestamp: float | None = None
    cooldown_until: float = 0.0  # monotonic seconds
    daily_reset_timestamp: float = 0.0  # epoch seconds
    last_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    last_used_timestamp: float = 0.0  # monotonic seconds
    total_tokens_used: int = 0

    def is_available(self, now_mono: float | None = None, now_epoch: float | None = None) -> bool:
        """Return True if model is available for immediate request dispatch."""
        if self.status == GeminiModelStatus.UNSUPPORTED:
            return False

        now_m = now_mono if now_mono is not None else time.monotonic()
        now_e = now_epoch if now_epoch is not None else time.time()

        # Check if RPD daily exhaustion has expired
        if self.status == GeminiModelStatus.EXHAUSTED_RPD:
            if self.daily_reset_timestamp > 0.0 and now_e >= self.daily_reset_timestamp:
                logger.info("Daily quota for model '%s' has reached reset time; restoring to HEALTHY.", self.model_id)
                self.status = GeminiModelStatus.HEALTHY
                self.consecutive_failures = 0
                self.daily_reset_timestamp = 0.0
                return True
            return False

        # Check if transient cooldown has expired
        if self.status in (GeminiModelStatus.COOLDOWN_RPM, GeminiModelStatus.PROJECT_LIMITED):
            if now_m >= self.cooldown_until:
                self.status = GeminiModelStatus.HEALTHY
                self.consecutive_failures = 0
                return True
            return False

        return self.status == GeminiModelStatus.HEALTHY


class GeminiModelPool:
    """
    Quota-aware model pool manager for Google Gemini models.
    Provides deterministic load distribution (least-recently-used among healthy models)
    and robust per-model rate limit / daily quota exhaustion tracking.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        primary_model: str | None = None,
        fallback_models: list[str] | None = None,
    ):
        configured_models: list[str] = []
        if models:
            configured_models = list(models)
        else:
            p_model = primary_model or "gemini-3.5-flash-lite"
            f_models = fallback_models if fallback_models is not None else ["gemini-3.1-flash-lite"]
            configured_models = [p_model] + [m for m in f_models if m != p_model]

        # Deduplicate while preserving order
        seen = set()
        self.models: list[str] = []
        for m in configured_models:
            clean = m.strip()
            if clean and clean not in seen:
                seen.add(clean)
                self.models.append(clean)

        if not self.models:
            self.models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

        self.models_health: dict[str, GeminiModelHealth] = {
            m: GeminiModelHealth(model_id=m) for m in self.models
        }
        self.project_cooldown_until: float = 0.0

    def is_project_limited(self) -> bool:
        """Return True if a project-wide limit is currently active."""
        return time.monotonic() < self.project_cooldown_until

    def get_candidate_models(self) -> list[str]:
        """
        Return candidate models ordered by priority and health.
        Healthy candidates are selected via Least-Recently-Used (LRU) for fair quota distribution.
        """
        now_mono = time.monotonic()
        now_epoch = time.time()

        available = [
            m for m in self.models
            if self.models_health[m].is_available(now_mono, now_epoch)
        ]

        if available:
            # Sort healthy models by least-recently-used (smallest last_used_timestamp)
            return sorted(available, key=lambda m: self.models_health[m].last_used_timestamp)

        # If none are strictly healthy, sort by earliest expected availability
        # (transient RPM cooldowns first, then RPD)
        def _availability_key(m: str) -> float:
            h = self.models_health[m]
            if h.status == GeminiModelStatus.UNSUPPORTED:
                return float("inf")
            if h.status == GeminiModelStatus.COOLDOWN_RPM:
                return h.cooldown_until
            if h.status == GeminiModelStatus.EXHAUSTED_RPD:
                return (h.daily_reset_timestamp - now_epoch) + now_mono
            return now_mono

        return sorted(self.models, key=_availability_key)

    def record_success(self, model_id: str, elapsed_ms: float, tokens: int | None = None) -> None:
        """Record successful completion on a model."""
        now_m = time.monotonic()
        h = self.models_health.get(model_id)
        if not h:
            h = GeminiModelHealth(model_id=model_id)
            self.models_health[model_id] = h

        h.status = GeminiModelStatus.HEALTHY
        h.consecutive_failures = 0
        h.successful_requests += 1
        h.total_requests += 1
        h.last_status_code = "200_OK"
        h.last_latency_ms = elapsed_ms
        h.total_latency_ms += elapsed_ms
        h.last_used_timestamp = now_m
        if tokens:
            h.total_tokens_used += tokens

    def record_failure(
        self,
        model_id: str,
        error_type: GeminiErrorType,
        retry_after: float | None = None,
        reason: str = "",
    ) -> None:
        """Record failure and apply model-specific or project-wide cooldowns."""
        now_m = time.monotonic()
        now_e = time.time()
        h = self.models_health.get(model_id)
        if not h:
            h = GeminiModelHealth(model_id=model_id)
            self.models_health[model_id] = h

        h.consecutive_failures += 1
        h.total_requests += 1
        h.last_failure_reason = reason
        h.last_failure_timestamp = now_e
        h.last_status_code = error_type.value

        if error_type == GeminiErrorType.RPD_EXHAUSTED:
            h.status = GeminiModelStatus.EXHAUSTED_RPD
            reset_ts = now_e + retry_after if retry_after and retry_after > 300.0 else get_pacific_midnight_timestamp()
            h.daily_reset_timestamp = reset_ts
            logger.warning(
                "Gemini model '%s' DAILY quota exhausted (RPD). Model parked until reset at %s (~%.1fh). Reason: %s",
                model_id,
                time.strftime("%Y-%m-%d %H:%M:%S PT", time.gmtime(reset_ts)),
                (reset_ts - now_e) / 3600.0,
                reason,
            )

        elif error_type == GeminiErrorType.RPM_TPM_LIMIT:
            h.status = GeminiModelStatus.COOLDOWN_RPM
            cooldown = retry_after if retry_after and 0.0 < retry_after <= 300.0 else 60.0
            h.cooldown_until = now_m + cooldown
            logger.warning(
                "Gemini model '%s' rate limited (RPM/TPM). Cooldown for %.1fs. Reason: %s",
                model_id,
                cooldown,
                reason,
            )

        elif error_type == GeminiErrorType.PROJECT_WIDE_LIMIT:
            h.status = GeminiModelStatus.PROJECT_LIMITED
            cooldown = retry_after if retry_after and retry_after > 0 else 60.0
            self.project_cooldown_until = now_m + cooldown
            logger.error(
                "Gemini project-wide quota/billing limit encountered. All Gemini models parked for %.1fs. Reason: %s",
                cooldown,
                reason,
            )

        elif error_type == GeminiErrorType.UNSUPPORTED_MODEL:
            h.status = GeminiModelStatus.UNSUPPORTED
            logger.error("Gemini model '%s' is not supported on this endpoint. Disabling permanently.", model_id)

        else:
            # Server/Network transient error: small backoff
            h.status = GeminiModelStatus.COOLDOWN_RPM
            h.cooldown_until = now_m + 5.0

    def get_pool_status(self) -> dict[str, Any]:
        """Return safe observability summary of the model pool."""
        now_mono = time.monotonic()
        now_epoch = time.time()
        return {
            "models": [
                {
                    "model_id": m,
                    "status": self.models_health[m].status.value,
                    "available": self.models_health[m].is_available(now_mono, now_epoch),
                    "total_requests": self.models_health[m].total_requests,
                    "successful_requests": self.models_health[m].successful_requests,
                    "consecutive_failures": self.models_health[m].consecutive_failures,
                    "last_latency_ms": self.models_health[m].last_latency_ms,
                    "last_error": self.models_health[m].last_failure_reason,
                    "daily_reset_in_hours": round(max(0.0, (self.models_health[m].daily_reset_timestamp - now_epoch) / 3600.0), 2)
                    if self.models_health[m].daily_reset_timestamp > 0 else 0.0,
                }
                for m in self.models
            ],
            "project_limited": self.is_project_limited(),
        }
