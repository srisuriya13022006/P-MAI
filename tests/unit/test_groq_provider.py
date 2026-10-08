from app.llm.groq_provider import GroqProvider


def test_groq_provider_uses_environment_api_key(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.responses = self

        def create(self, **kwargs):
            captured["create"] = kwargs
            class FakeResponse:
                output_text = "cloud answer"
            return FakeResponse()

    monkeypatch.setattr("app.llm.groq_provider.settings.groq_api_key", "test-key", raising=False)
    monkeypatch.setattr("app.llm.groq_provider.OpenAI", FakeClient)

    provider = GroqProvider(model="openai/gpt-oss-20b")
    result = provider.generate([
        {"role": "user", "content": "hello"},
    ], system_prompt="You are helpful.")

    assert result == "cloud answer"
    assert captured["kwargs"]["api_key"] == "test-key"
    assert captured["kwargs"]["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["create"]["model"] == "openai/gpt-oss-20b"
