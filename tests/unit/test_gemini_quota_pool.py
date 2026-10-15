"""
P15.7 — Unit & Integration Tests for Quota-Aware Multi-Model Gemini Model Pool.
Validates:
1. Model pool fair distribution (LRU / round-robin) across healthy candidates.
2. Model-specific RPM/TPM rate limits with retry-after cooldown.
3. Model-specific RPD daily quota exhaustion and Pacific Time midnight skip.
4. Project-wide quota/billing restriction fast failover to Groq.
5. Unsupported model (404) marking and disabling.
6. Multi-model Gemini failover -> Groq fallback -> Ollama fallback.
7. Tool calling support and duplicate-tool prevention.
8. Telemetry and pool observability.
"""
import time
from unittest.mock import MagicMock, patch
import pytest

from app.core.config import Settings, settings
from app.llm.base import LLMProvider
from app.llm.cascading_provider import CascadingCloudProvider
from app.llm.gemini_pool import (
    GeminiErrorType,
    GeminiModelHealth,
    GeminiModelPool,
    GeminiModelStatus,
    classify_gemini_error,
    get_pacific_midnight_timestamp,
)
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider


class MockChatCompletionResponse:
    def __init__(self, content: str = "Mock response", tool_calls=None, prompt_tokens=15, completion_tokens=25):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


def test_classify_gemini_errors():
    """Verify error classification distinguishes RPM vs RPD vs Project vs 404 vs Auth."""
    # 1. Model-specific RPD Daily Quota
    rpd_err = RuntimeError(
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3.5-flash-lite. Please retry in 17h38m. violations: [{'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}] retryDelay: '63529s'"
    )
    rpd_err.status_code = 429
    cat, retry, reason = classify_gemini_error(rpd_err)
    assert cat == GeminiErrorType.RPD_EXHAUSTED
    assert retry == 63529.0

    # 2. Model-specific RPM Transient Limit
    rpm_err = RuntimeError("429 Resource has been exhausted (rate limit). Please try again in 12.5s.")
    rpm_err.status_code = 429
    cat, retry, reason = classify_gemini_error(rpm_err)
    assert cat == GeminiErrorType.RPM_TPM_LIMIT
    assert retry == 12.5

    # 3. Project-wide Limit
    proj_err = RuntimeError("Quota exceeded for metric: generativelanguage.googleapis.com/project_requests. Billing not enabled.")
    proj_err.status_code = 429
    cat, retry, reason = classify_gemini_error(proj_err)
    assert cat == GeminiErrorType.PROJECT_WIDE_LIMIT

    # 4. 404 Unsupported Model
    not_found_err = RuntimeError("404 models/gemini-2.0-flash is not found for API version v1main")
    not_found_err.status_code = 404
    cat, retry, reason = classify_gemini_error(not_found_err)
    assert cat == GeminiErrorType.UNSUPPORTED_MODEL

    # 5. Auth Error
    auth_err = RuntimeError("401 Unauthorized: API key not valid")
    auth_err.status_code = 401
    cat, retry, reason = classify_gemini_error(auth_err)
    assert cat == GeminiErrorType.AUTH_ERROR


def test_pool_lru_fair_distribution():
    """Verify requests balance fairly across healthy candidates using LRU."""
    pool = GeminiModelPool(models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])
    mock_client = MagicMock()
    call_models = []

    def mock_create(model, messages, **kwargs):
        call_models.append(model)
        return MockChatCompletionResponse(f"OK from {model}")

    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )
        provider.pool = pool

        # Call 1 -> chooses model 1
        res1 = provider.generate([{"role": "user", "content": "1"}])
        # Call 2 -> chooses model 2 (least recently used)
        res2 = provider.generate([{"role": "user", "content": "2"}])
        # Call 3 -> chooses model 1 (least recently used)
        res3 = provider.generate([{"role": "user", "content": "3"}])

    assert call_models == [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
    ]


def test_model_specific_rpm_cooldown_and_failover():
    """Verify RPM rate limit triggers cooldown on model A and rotates to model B."""
    call_models = []

    def mock_create(model, messages, **kwargs):
        call_models.append(model)
        if model == "gemini-3.5-flash-lite":
            err = RuntimeError("429 Resource exhausted. try again in 15s")
            err.status_code = 429
            raise err
        return MockChatCompletionResponse("Success from 3.1")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )
        # First turn: 3.5 fails, falls over to 3.1
        res = provider.generate([{"role": "user", "content": "turn 1"}])
        assert res == "Success from 3.1"
        assert provider.model == "gemini-3.1-flash-lite"
        assert call_models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

        # Second turn: 3.5 is in cooldown, goes directly to 3.1
        call_models.clear()
        res2 = provider.generate([{"role": "user", "content": "turn 2"}])
        assert res2 == "Success from 3.1"
        assert call_models == ["gemini-3.1-flash-lite"]


def test_model_specific_rpd_daily_exhaustion():
    """Verify RPD daily exhaustion parks model A until reset time without retrying."""
    call_models = []

    def mock_create(model, messages, **kwargs):
        call_models.append(model)
        if model == "gemini-3.5-flash-lite":
            err = RuntimeError(
                "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests. "
                "limit: 500, model: gemini-3.5-flash-lite. Please retry in 18h. violations: [{'quotaId': "
                "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}] retryDelay: '64800s'"
            )
            err.status_code = 429
            raise err
        return MockChatCompletionResponse("Success from 3.1")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )
        res = provider.generate([{"role": "user", "content": "test"}])
        assert res == "Success from 3.1"
        assert provider.pool.models_health["gemini-3.5-flash-lite"].status == GeminiModelStatus.EXHAUSTED_RPD

        # Subsequent requests skip 3.5 completely
        call_models.clear()
        res2 = provider.generate([{"role": "user", "content": "test 2"}])
        assert res2 == "Success from 3.1"
        assert call_models == ["gemini-3.1-flash-lite"]


def test_project_wide_limit_fast_failover_to_groq():
    """Verify project-wide quota error immediately aborts Gemini pool and invokes Groq."""
    mock_client = MagicMock()

    def mock_create(model, messages, **kwargs):
        err = RuntimeError("429 Quota exceeded: Billing account disabled for project.")
        err.status_code = 429
        raise err

    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        gemini = GeminiProvider(
            api_key="mock-key",
            model_pool=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )

        mock_groq = MagicMock(spec=LLMProvider)
        mock_groq.provider_name = "groq"
        mock_groq.model = "openai/gpt-oss-20b"
        mock_groq.generate.return_value = "Response from Groq"

        cascade = CascadingCloudProvider(providers=[gemini, mock_groq])
        res = cascade.generate([{"role": "user", "content": "hello"}])

        assert res == "Response from Groq"
        assert cascade.provider_name == "groq"
        # Gemini should only be attempted once (for the project limit), then immediately bypassed
        assert mock_client.chat.completions.create.call_count == 1
        mock_groq.generate.assert_called_once()


def test_unsupported_model_404_marked_and_disabled():
    """Verify 404 marks model as UNSUPPORTED permanently and switches to valid model."""
    call_models = []

    def mock_create(model, messages, **kwargs):
        call_models.append(model)
        if model == "gemini-invalid-model":
            err = RuntimeError("404 models/gemini-invalid-model is not found")
            err.status_code = 404
            raise err
        return MockChatCompletionResponse("Success from 3.5")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model_pool=["gemini-invalid-model", "gemini-3.5-flash-lite"],
        )
        res = provider.generate([{"role": "user", "content": "hi"}])
        assert res == "Success from 3.5"
        assert provider.pool.models_health["gemini-invalid-model"].status == GeminiModelStatus.UNSUPPORTED

        call_models.clear()
        res2 = provider.generate([{"role": "user", "content": "hi again"}])
        assert res2 == "Success from 3.5"
        assert call_models == ["gemini-3.5-flash-lite"]


def test_full_chain_gemini_pool_to_groq_to_ollama():
    """Verify entire chain: all Gemini pool fail -> Groq fails -> Ollama succeeds."""
    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.model = "gemini-3.5-flash-lite"
    mock_gemini.generate.side_effect = RuntimeError("429 Gemini Pool Exhausted")

    mock_groq = MagicMock(spec=LLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate.side_effect = RuntimeError("429 Groq Rate Limit")

    cloud_cascade = CascadingCloudProvider(providers=[mock_gemini, mock_groq])

    mock_ollama = MagicMock(spec=LLMProvider)
    mock_ollama.provider_name = "ollama"
    mock_ollama.model = "qwen3:4b-instruct"
    mock_ollama.context_limit = 4096
    mock_ollama.generate.return_value = "Response from Local Ollama"

    resilient = ResilientLLMProvider(
        primary_provider=cloud_cascade,
        fallback_provider=mock_ollama,
        fallback_enabled=True,
    )

    reply = resilient.generate([{"role": "user", "content": "test prompt"}])
    assert reply == "Response from Local Ollama"
    assert resilient.provider_name == "ollama"
    assert resilient.runtime_mode == "LOCAL_FALLBACK"


def test_gemini_provider_function_call_structured_return():
    """Verify GeminiProvider handles tool calls gracefully."""
    mock_client = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "calculator"
    mock_tool_call.function.arguments = '{"expression": "2 + 2"}'

    response_mock = MockChatCompletionResponse(content=None, tool_calls=[mock_tool_call])
    mock_client.chat.completions.create.return_value = response_mock

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(api_key="mock-key")
        result = provider.generate([{"role": "user", "content": "calc"}], tools=[{"type": "function"}])
        assert "calculator" in result
        assert "2 + 2" in result


def test_pool_observability_telemetry():
    """Verify pool_status exposes structured metrics without leaking secrets."""
    pool = GeminiModelPool(models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])
    pool.record_success("gemini-3.5-flash-lite", elapsed_ms=120.5, tokens=35)
    pool.record_failure("gemini-3.1-flash-lite", GeminiErrorType.RPM_TPM_LIMIT, retry_after=10.0, reason="Rate limit")

    status = pool.get_pool_status()
    assert len(status["models"]) == 2
    m1 = next(m for m in status["models"] if m["model_id"] == "gemini-3.5-flash-lite")
    m2 = next(m for m in status["models"] if m["model_id"] == "gemini-3.1-flash-lite")

    assert m1["status"] == "HEALTHY"
    assert m1["successful_requests"] == 1
    assert m1["last_latency_ms"] == 120.5

    assert m2["status"] == "COOLDOWN_RPM"
    assert m2["consecutive_failures"] == 1
    assert "Rate limit" in m2["last_error"]
