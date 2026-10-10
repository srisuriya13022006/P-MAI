"""
P15.4 — Runtime Modes and Safe Observability.
Defines runtime operation modes (FULL_CLOUD, DEGRADED, OFFLINE) and latency metadata tracking.
Strictly ensures zero API keys, auth headers, or raw secrets are ever exposed.
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class RuntimeMode(str, Enum):
    FULL_CLOUD = "FULL_CLOUD"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class CloudErrorCategory(str, Enum):
    RECOVERABLE = "RECOVERABLE"      # Network, Timeout, Connection failure -> local fallback
    HARD_CONFIG = "HARD_CONFIG"      # 401, 403, Authentication, Missing/Invalid API key -> deterministic error, no silent fallback
    RATE_LIMIT = "RATE_LIMIT"        # 429 -> controlled fallback if permitted
    UNRECOVERABLE = "UNRECOVERABLE"  # Hard logic/schema errors


def classify_cloud_error(exc: Exception) -> CloudErrorCategory:
    """
    Classify cloud error according to P15.4 policy:
    - HARD_CONFIG: 401 Unauthorized, 403 Forbidden, AuthenticationFailure, InvalidApiKey -> NEVER silently fall back.
    - RECOVERABLE: Network failure, timeout, connection drop, transient 502/503 -> controlled local fallback.
    - RATE_LIMIT: 429 Rate limit -> controlled fallback with surfaced reason.
    - UNRECOVERABLE: Hard logic/schema errors.
    """
    err_str = str(exc).lower()
    err_type = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)

    # 1. Hard Configuration / Authentication Errors
    if status_code in (401, 403):
        return CloudErrorCategory.HARD_CONFIG
    if any(k in err_str for k in (
        "401", "403", "unauthorized", "authentication failed", "authentication_failure",
        "invalid api key", "invalid_api_key", "invalid_configuration", "invalid configuration",
        "forbidden", "api key is not configured", "invalid apikey"
    )):
        return CloudErrorCategory.HARD_CONFIG
    if any(k in err_type for k in ("auth", "permission", "invalidapikey", "configuration", "config")):
        return CloudErrorCategory.HARD_CONFIG

    # 2. Rate Limiting
    if status_code == 429 or "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
        return CloudErrorCategory.RATE_LIMIT

    # 3. Recoverable Network & Timeout Failures
    if any(k in err_type for k in ("connect", "timeout", "network", "socket", "unavailable")):
        return CloudErrorCategory.RECOVERABLE
    if any(k in err_str for k in (
        "timeout", "timed out", "connection", "unreachable", "network",
        "econnreset", "econnrefused", "502", "503", "504",
        "connection_failure", "network_failure"
    )):
        return CloudErrorCategory.RECOVERABLE

    return CloudErrorCategory.RECOVERABLE



def determine_runtime_mode(stt_mode: str, llm_mode: str) -> RuntimeMode:
    """
    Determine composite system runtime mode from STT and LLM component states:
    - FULL_CLOUD: Both STT and LLM in CLOUD mode
    - DEGRADED: One cloud, one local/fallback
    - OFFLINE: Both in LOCAL/OFFLINE mode
    """
    is_stt_cloud = stt_mode.upper() == "CLOUD"
    is_llm_cloud = llm_mode.upper() == "CLOUD"

    if is_stt_cloud and is_llm_cloud:
        return RuntimeMode.FULL_CLOUD
    if not is_stt_cloud and not is_llm_cloud:
        return RuntimeMode.OFFLINE
    return RuntimeMode.DEGRADED


@dataclass
class SafeExecutionMetadata:
    """
    Structured execution observability metadata.
    Sanitized against credential exposure.
    """
    stt_provider: str = "assemblyai"
    stt_mode: str = "CLOUD"
    llm_provider: str = "groq"
    llm_mode: str = "CLOUD"
    runtime_mode: RuntimeMode = RuntimeMode.FULL_CLOUD
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stt_provider": self.stt_provider,
            "stt_mode": self.stt_mode,
            "llm_provider": self.llm_provider,
            "llm_mode": self.llm_mode,
            "runtime_mode": self.runtime_mode.value,
            "latency": {
                "stt_latency_ms": round(self.stt_latency_ms, 2),
                "llm_latency_ms": round(self.llm_latency_ms, 2),
                "tool_latency_ms": round(self.tool_latency_ms, 2),
                "total_latency_ms": round(self.total_latency_ms, 2),
            },
        }
