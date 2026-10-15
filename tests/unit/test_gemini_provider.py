"""
P15.5 — Unit Tests for Gemini Cloud LLM Provider and Cascading Model Switching.
Validates:
1. GeminiProvider initialization, key resolution, and credential leakage prevention.
2. GeminiProvider successful generation and token usage tracking.
3. GeminiProvider multi-model failover on rate limits (429 cooldowns).
4. CascadingCloudProvider: Gemini primary -> switches to Groq upon ANY Gemini error.
5. ResilientLLMProvider: Gemini fails -> Groq fails -> Local Ollama fallback.
6. Factory configuration: Gemini is primary cloud model when configured.
"""
from unittest.mock import MagicMock, patch
import pytest

from app.core.config import Settings, settings
from app.llm.base import LLMProvider
from app.llm.cascading_provider import CascadingCloudProvider
from app.llm.factory import create_cloud_provider, create_llm_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider


# -------------------------------------------------------------------------
# Helpers / Mocks
# -------------------------------------------------------------------------

class MockChatCompletionResponse:
    def __init__(self, content: str = "Test response from cloud", prompt_tokens: int = 20, completion_tokens: int = 30):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)


# -------------------------------------------------------------------------
# 1. GeminiProvider Initialization & Key Resolution
# -------------------------------------------------------------------------

def test_gemini_provider_missing_key_raises():
    """Verify GeminiProvider raises clear ValueError when no key is configured."""
    with patch("app.core.config.settings.gemini_api_key", None), \
         patch("app.core.config.settings.google_api_key", None), \
         patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError) as exc:
            GeminiProvider(api_key=None)
        assert "GEMINI_API_KEY" in str(exc.value)


def test_gemini_provider_resolves_google_api_key_alias():
    """Verify GeminiProvider recognizes GOOGLE_API_KEY as an alias."""
    mock_client = MagicMock()
    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client) as mock_cls, \
         patch.object(settings, "gemini_api_key", None), \
         patch.object(settings, "google_api_key", None), \
         patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-google-key", "GEMINI_API_KEY": ""}):
        provider = GeminiProvider(api_key=None, fallback_models=["gemini-1.5-flash"])
        assert provider.primary_model == "gemini-2.5-flash"
        mock_cls.assert_called_once()
        assert mock_cls.call_args[1]["api_key"] == "mock-google-key"


def test_gemini_credential_leakage_prevention():
    """Verify secret API keys are never leaked in string representation."""
    secret_key = "AIzaSySecretGeminiKey12345"
    mock_client = MagicMock()
    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(api_key=secret_key)
        assert secret_key not in str(provider)
        assert secret_key not in repr(provider)


# -------------------------------------------------------------------------
# 2. GeminiProvider Successful Generation
# -------------------------------------------------------------------------

def test_gemini_provider_generate_success():
    """Verify GeminiProvider generates response and captures token metrics."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MockChatCompletionResponse(
        content="Hello from Gemini!",
        prompt_tokens=25,
        completion_tokens=40,
    )

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(api_key="mock-key", model="gemini-2.5-flash")
        res = provider.generate([{"role": "user", "content": "Hello"}])

    assert res == "Hello from Gemini!"
    assert provider.provider_name == "gemini"
    assert provider.model == "gemini-2.5-flash"
    assert provider.last_usage["prompt_tokens"] == 25
    assert provider.last_usage["completion_tokens"] == 40
    assert provider.last_status == "200_OK"


# -------------------------------------------------------------------------
# 3. GeminiProvider Internal Multi-Model Failover
# -------------------------------------------------------------------------

def test_gemini_provider_model_failover_on_429():
    """Verify GeminiProvider switches from primary model to fallback model upon 429."""
    call_history = []

    def mock_create(model, messages, **kwargs):
        call_history.append(model)
        if model == "gemini-2.5-flash":
            err = RuntimeError("429 Resource has been exhausted (rate limit). Please try again in 30s.")
            err.response = MagicMock(headers={"retry-after": "30"})
            raise err
        elif model == "gemini-1.5-flash":
            return MockChatCompletionResponse("Response from Gemini 1.5 Flash")
        raise RuntimeError(f"Unexpected model: {model}")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = mock_create

    with patch("app.llm.gemini_provider.OpenAI", return_value=mock_client):
        provider = GeminiProvider(
            api_key="mock-key",
            model="gemini-2.5-flash",
            fallback_models=["gemini-1.5-flash"],
        )
        res = provider.generate([{"role": "user", "content": "test"}])

    assert res == "Response from Gemini 1.5 Flash"
    assert provider.model == "gemini-1.5-flash"
    assert call_history == ["gemini-2.5-flash", "gemini-1.5-flash"]

    # Subsequent call skips 2.5-flash while in cooldown
    call_history.clear()
    res2 = provider.generate([{"role": "user", "content": "second turn"}])
    assert res2 == "Response from Gemini 1.5 Flash"
    assert call_history == ["gemini-1.5-flash"]


# -------------------------------------------------------------------------
# 4. CascadingCloudProvider: Gemini Primary -> Switches to Groq
# -------------------------------------------------------------------------

def test_cascading_cloud_provider_gemini_primary_success():
    """Verify Gemini as primary fulfills request without invoking Groq switch."""
    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.model = "gemini-2.5-flash"
    mock_gemini.generate.return_value = "Response from Primary Gemini"

    mock_groq = MagicMock(spec=LLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"

    cascade = CascadingCloudProvider(providers=[mock_gemini, mock_groq])
    res = cascade.generate([{"role": "user", "content": "hi"}])

    assert res == "Response from Primary Gemini"
    assert cascade.provider_name == "gemini"
    mock_gemini.generate.assert_called_once()
    mock_groq.generate.assert_not_called()
    assert cascade.last_failover_occurred is False


def test_cascading_cloud_provider_gemini_error_switches_to_groq():
    """Verify when Gemini raises ANY error, CascadingCloudProvider immediately switches to Groq."""
    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.model = "gemini-2.5-flash"
    mock_gemini.generate.side_effect = RuntimeError("Gemini 503 Overloaded or Quota Exceeded")

    mock_groq = MagicMock(spec=LLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.model = "openai/gpt-oss-120b"
    mock_groq.generate.return_value = "Response from Groq switch model"

    cascade = CascadingCloudProvider(providers=[mock_gemini, mock_groq])
    res = cascade.generate([{"role": "user", "content": "hi"}])

    assert res == "Response from Groq switch model"
    assert cascade.provider_name == "groq"
    assert cascade.last_failover_occurred is True
    assert "gemini" in cascade.last_failover_reason.lower()

    mock_gemini.generate.assert_called_once()
    mock_groq.generate.assert_called_once()


def test_cascading_cloud_provider_all_fail_raises():
    """Verify when both Gemini and Groq fail, the exception is raised for local fallback."""
    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate.side_effect = RuntimeError("Gemini failed")

    mock_groq = MagicMock(spec=LLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.generate.side_effect = RuntimeError("Groq rate limited")

    cascade = CascadingCloudProvider(providers=[mock_gemini, mock_groq])

    with pytest.raises(RuntimeError) as exc:
        cascade.generate([{"role": "user", "content": "hi"}])
    assert "Groq rate limited" in str(exc.value)


# -------------------------------------------------------------------------
# 5. ResilientLLMProvider: Cloud Cascade -> Local Ollama Fallback
# -------------------------------------------------------------------------

def test_resilient_llm_provider_cloud_cascade_to_ollama_fallback():
    """Verify Gemini error -> Groq error -> smoothly falls back to Local Ollama."""
    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.provider_name = "gemini"
    mock_gemini.generate.side_effect = RuntimeError("Gemini down")

    mock_groq = MagicMock(spec=LLMProvider)
    mock_groq.provider_name = "groq"
    mock_groq.generate.side_effect = RuntimeError("Groq 429 rate limit")

    cascade = CascadingCloudProvider(providers=[mock_gemini, mock_groq])

    mock_ollama = MagicMock(spec=LLMProvider)
    mock_ollama.provider_name = "ollama"
    mock_ollama.model = "qwen3:4b-instruct"
    mock_ollama.generate.return_value = "Response from Local Qwen model"

    resilient = ResilientLLMProvider(
        primary_provider=cascade,
        fallback_provider=mock_ollama,
        fallback_enabled=True,
    )

    res = resilient.generate([{"role": "user", "content": "explain quantum computing"}])

    assert res == "Response from Local Qwen model"
    assert resilient.last_provider_used == "ollama"
    assert resilient.runtime_mode == "LOCAL_FALLBACK"


# -------------------------------------------------------------------------
# 6. Factory Dispatch: Gemini Primary and Groq Switch
# -------------------------------------------------------------------------

def test_create_cloud_provider_gemini_primary_when_both_keys_present():
    """Verify create_cloud_provider orders Gemini first and Groq second when both keys exist."""
    cfg = Settings(
        database_url="postgresql+psycopg://u:p@localhost/db",
        gemini_api_key="mock-gemini-key",
        groq_api_key="mock-groq-key",
        cloud_llm_provider="auto",
    )

    with patch("app.llm.gemini_provider.OpenAI"), patch("app.llm.groq_provider.OpenAI"):
        cloud_prov = create_cloud_provider(cfg)

    assert isinstance(cloud_prov, CascadingCloudProvider)
    assert len(cloud_prov.providers) == 2
    # Gemini is the PRIMARY model
    assert isinstance(cloud_prov.providers[0], GeminiProvider)
    # Groq is the SWITCH/FAILOVER model
    assert isinstance(cloud_prov.providers[1], GroqProvider)


def test_create_cloud_provider_only_gemini_key():
    """Verify create_cloud_provider returns GeminiProvider when only Gemini key is present."""
    cfg = Settings(
        database_url="postgresql+psycopg://u:p@localhost/db",
        gemini_api_key="mock-gemini-key",
        groq_api_key=None,
    )

    with patch("app.llm.gemini_provider.OpenAI"):
        cloud_prov = create_cloud_provider(cfg)

    assert isinstance(cloud_prov, GeminiProvider)
    assert cloud_prov.provider_name == "gemini"


def test_create_llm_provider_cloud_wraps_in_resilient():
    """Verify create_llm_provider creates ResilientLLMProvider with Gemini primary and Ollama fallback."""
    cfg = Settings(
        database_url="postgresql+psycopg://u:p@localhost/db",
        gemini_api_key="mock-gemini-key",
        groq_api_key="mock-groq-key",
        llm_provider="cloud",
        llm_fallback_enabled=True,
    )

    with patch("app.llm.gemini_provider.OpenAI"), patch("app.llm.groq_provider.OpenAI"):
        provider = create_llm_provider(settings_obj=cfg)

    assert isinstance(provider, ResilientLLMProvider)
    assert isinstance(provider.primary_provider, CascadingCloudProvider)
    assert isinstance(provider.fallback_provider, OllamaProvider)
