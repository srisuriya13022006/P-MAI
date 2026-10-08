from app.memory.embeddings import OllamaEmbeddingProvider, LocalHashEmbeddingProvider


def test_ollama_embedding_provider_uses_model_response(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self.embeddings = [[0.1, 0.2, 0.3]]

    def fake_embed(model, input, **kwargs):
        captured["model"] = model
        captured["input"] = input
        return FakeResponse()

    monkeypatch.setattr("ollama.embed", fake_embed)
    provider = OllamaEmbeddingProvider(model="nomic-embed-text")

    vector = provider.embed("hello world")

    assert captured["model"] == "nomic-embed-text"
    assert captured["input"] == "hello world"
    assert vector == [0.1, 0.2, 0.3]


def test_local_hash_embedding_provider_returns_numeric_vector():
    provider = LocalHashEmbeddingProvider()
    vector = provider.embed("likes coffee")

    assert len(vector) == provider.dimensions
    assert all(isinstance(value, float) for value in vector)
