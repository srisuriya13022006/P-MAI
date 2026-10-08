import hashlib
import math
import re
from typing import Any

import ollama


class EmbeddingProvider:
    """Embedding contract for semantic memory retrieval."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free vectors for local development and tests."""

    dimensions = 128

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Semantic embedding provider backed by an Ollama model when available."""

    def __init__(self, model: str, *, host: str | None = None, dimensions: int | None = None) -> None:
        self.model = model
        self.host = host
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        response = ollama.embed(
            model=self.model,
            input=text,
            truncate=True,
            dimensions=self.dimensions,
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise ValueError("Ollama embedding response did not include any vectors.")
        return list(embeddings[0])


class EmbeddingProviderFactory:
    """Creates the preferred embedding provider while preserving a local fallback."""

    @staticmethod
    def create(model_name: str | None = None, host: str | None = None) -> EmbeddingProvider:
        preferred_model = model_name or "nomic-embed-text"
        try:
            provider = OllamaEmbeddingProvider(model=preferred_model, host=host)
            provider.embed("health-check")
            return provider
        except Exception:
            return LocalHashEmbeddingProvider()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        shorter = min(len(left), len(right))
        left = left[:shorter]
        right = right[:shorter]
    return sum(a * b for a, b in zip(left, right, strict=False))