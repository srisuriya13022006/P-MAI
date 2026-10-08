from datetime import datetime
import re

from app.database.models import Memory
from app.memory.embeddings import EmbeddingProvider, LocalHashEmbeddingProvider, cosine_similarity
from app.memory.temporal import calculate_temporal_score, is_expired

# Configurable retrieval weights
DEFAULT_WEIGHT_LEXICAL: float = 1.0
DEFAULT_WEIGHT_SEMANTIC: float = 1.0
DEFAULT_WEIGHT_IMPORTANCE: float = 0.5
DEFAULT_WEIGHT_TEMPORAL: float = 0.5


class MemoryRetriever:
    """Ranks a user's memories using hybrid lexical, semantic, importance, and temporal scoring."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        weight_lexical: float = DEFAULT_WEIGHT_LEXICAL,
        weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
        weight_importance: float = DEFAULT_WEIGHT_IMPORTANCE,
        weight_temporal: float = DEFAULT_WEIGHT_TEMPORAL,
    ) -> None:
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.weight_lexical = weight_lexical
        self.weight_semantic = weight_semantic
        self.weight_importance = weight_importance
        self.weight_temporal = weight_temporal

    def retrieve(
        self,
        query: str,
        memories: list[Memory],
        limit: int = 5,
        now: datetime | None = None,
    ) -> list[Memory]:
        query_tokens = self._tokens(query)
        query_embedding = self.embedding_provider.embed(query)
        ranked = []

        for memory in memories:
            # 1. Skip inactive and expired memories
            if not getattr(memory, "is_active", True):
                continue
            if is_expired(memory, now):
                continue

            lexical_score = len(query_tokens & self._tokens(memory.content))
            embedding = memory.embedding or self.embedding_provider.embed(memory.content)
            semantic_score = cosine_similarity(query_embedding, embedding)

            # A memory must have non-zero relevance to the query
            if lexical_score + semantic_score <= 0:
                continue

            temporal_score = calculate_temporal_score(memory, now=now)
            normalized_importance = max(1.0, min(3.0, float(memory.importance))) / 3.0

            composite_score = (
                self.weight_lexical * lexical_score
                + self.weight_semantic * semantic_score
                + self.weight_importance * normalized_importance
                + self.weight_temporal * temporal_score
            )

            ranked.append((composite_score, memory.importance, temporal_score, memory))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in ranked[:limit]]

    def _tokens(self, value: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
        return {self._stem(token) for token in tokens}

    def _stem(self, token: str) -> str:
        normalized = token.lower()
        if len(normalized) <= 3:
            return normalized

        if normalized.endswith("ies"):
            return normalized[:-3] + "y"

        if normalized.endswith(("sses", "xes", "zes", "ches", "shes")):
            return normalized[:-2]

        if normalized.endswith("ing") and len(normalized) > 6:
            candidate = normalized[:-3]
            if len(candidate) >= 3:
                return candidate

        if normalized.endswith("s") and not normalized.endswith(("ss", "us", "is")) and len(normalized) > 4:
            return normalized[:-1]

        return normalized