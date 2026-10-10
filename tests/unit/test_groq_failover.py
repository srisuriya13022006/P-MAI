import time
from unittest.mock import MagicMock
from app.llm.groq_provider import GroqProvider


def test_groq_provider_multi_model_failover(monkeypatch):
    """Verify GroqProvider seamlessly switches to fallback model upon encountering 429 rate limit."""
    monkeypatch.setattr("app.llm.groq_provider.settings.groq_api_key", "mock-groq-key", raising=False)

    call_history = []

    class MockCompletion:
        def __init__(self, content):
            self.choices = [MagicMock(message=MagicMock(content=content))]
            self.usage = MagicMock(prompt_tokens=15, completion_tokens=25, total_tokens=40)

    class MockChatCompletions:
        def create(self, model, messages, **kwargs):
            call_history.append(model)
            if model == "openai/gpt-oss-20b":
                # Simulate 429 rate limit error
                err = RuntimeError("Rate limit reached on TPM. Please try again in 300s.")
                err.response = MagicMock(headers={"retry-after": "300"})
                raise err
            elif model == "openai/gpt-oss-120b":
                return MockCompletion("Answer from 120B model")
            raise RuntimeError(f"Unexpected model {model}")

    class MockClient:
        def __init__(self, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MockChatCompletions()

    monkeypatch.setattr("app.llm.groq_provider.OpenAI", MockClient)

    provider = GroqProvider(
        model="openai/gpt-oss-20b",
        fallback_models=["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
    )

    # First call: primary 20b fails with 429, falls over to 120b
    res = provider.generate([{"role": "user", "content": "hello"}])

    assert res == "Answer from 120B model"
    assert provider.model == "openai/gpt-oss-120b"
    assert provider.last_used_model == "openai/gpt-oss-120b"
    assert call_history == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    # Second call: primary 20b is in active cooldown, so provider skips 20b directly to 120b
    call_history.clear()
    res2 = provider.generate([{"role": "user", "content": "second question"}])

    assert res2 == "Answer from 120B model"
    assert call_history == ["openai/gpt-oss-120b"]  # Zero wasted 429 attempts!


def test_groq_provider_all_models_fail(monkeypatch):
    """Verify that when all cloud models fail, the error is propagated so ResilientLLMProvider triggers local fallback."""
    monkeypatch.setattr("app.llm.groq_provider.settings.groq_api_key", "mock-groq-key", raising=False)

    class MockChatCompletions:
        def create(self, model, messages, **kwargs):
            err = RuntimeError("Rate limit reached on TPM. Please try again in 60s.")
            err.response = MagicMock(headers={"retry-after": "60"})
            raise err

    class MockClient:
        def __init__(self, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MockChatCompletions()

    monkeypatch.setattr("app.llm.groq_provider.OpenAI", MockClient)

    provider = GroqProvider(
        model="openai/gpt-oss-20b",
        fallback_models=["openai/gpt-oss-120b"],
    )

    import pytest
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate([{"role": "user", "content": "hello"}])

    assert "Rate limit" in str(exc_info.value)
    assert provider.last_status == "429_RATE_LIMIT"
