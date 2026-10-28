"""
P16 — Procedural Skill Discovery & Retrieval.
Reuses existing LocalHashEmbeddingProvider and hybrid lexical/semantic scoring
without external LLM calls or additional vector databases.
"""
from dataclasses import dataclass
import re

from app.database.models import Skill
from app.memory.embeddings import EmbeddingProvider, LocalHashEmbeddingProvider, cosine_similarity


@dataclass
class SkillSearchResult:
    """Represents a scored skill match from discovery retrieval."""
    skill: Skill
    score: float
    is_exact_match: bool = False


@dataclass
class SkillRetrievalOutcome:
    """Outcome of skill resolution including ambiguity detection."""
    selected_skill: Skill | None = None
    is_ambiguous: bool = False
    ambiguous_candidates: list[Skill] = None

    def __post_init__(self):
        if self.ambiguous_candidates is None:
            self.ambiguous_candidates = []


class SkillRetriever:
    """Discovers and ranks saved procedural skills for an authorized owner."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        weight_lexical: float = 1.0,
        weight_semantic: float = 1.0,
    ) -> None:
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.weight_lexical = weight_lexical
        self.weight_semantic = weight_semantic

    def _tokens(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        tokens = set()
        for w in words:
            if len(w) > 3 and w.endswith("ing"):
                w = w[:-3]
            elif len(w) > 3 and w.endswith("s"):
                w = w[:-1]
            tokens.add(w)
        return tokens

    def resolve_skill(
        self,
        query: str,
        skills: list[Skill],
        ambiguity_margin: float = 0.15,
    ) -> SkillRetrievalOutcome:
        """
        Resolve a user query to a single skill or detect ambiguity.
        Never crosses user boundaries (caller passes only the owner's skills).
        """
        if not skills or not query.strip():
            return SkillRetrievalOutcome()

        q_clean = query.strip().lower()

        # 1. Exact Name Match (case-insensitive)
        for s in skills:
            if s.name.lower() == q_clean:
                return SkillRetrievalOutcome(selected_skill=s)

        # 2. Name containment: if user asks "Use my Research Brief Builder skill"
        name_matches = []
        for s in skills:
            if s.name.lower() in q_clean or q_clean in s.name.lower():
                name_matches.append(s)

        if len(name_matches) == 1:
            return SkillRetrievalOutcome(selected_skill=name_matches[0])
        elif len(name_matches) > 1:
            return SkillRetrievalOutcome(
                is_ambiguous=True,
                ambiguous_candidates=name_matches,
            )

        # 3. Hybrid Lexical + Semantic Search
        q_tokens = self._tokens(query)
        q_emb = self.embedding_provider.embed(query)
        scored: list[tuple[float, Skill]] = []

        for s in skills:
            corpus = f"{s.name} {s.description} {s.intended_use or ''}"
            c_tokens = self._tokens(corpus)
            overlap = len(q_tokens & c_tokens)
            lex_score = overlap / max(1, len(q_tokens))

            c_emb = self.embedding_provider.embed(corpus)
            sem_score = cosine_similarity(q_emb, c_emb)

            composite = (self.weight_lexical * lex_score) + (self.weight_semantic * sem_score)
            if composite > 0.2:
                scored.append((composite, s))

        if not scored:
            return SkillRetrievalOutcome()

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_skill = scored[0]

        # Check for ambiguity among top candidates
        if len(scored) > 1:
            second_score, second_skill = scored[1]
            if (top_score - second_score) <= ambiguity_margin and second_score >= 0.4:
                return SkillRetrievalOutcome(
                    is_ambiguous=True,
                    ambiguous_candidates=[top_skill, second_skill],
                )

        return SkillRetrievalOutcome(selected_skill=top_skill)
