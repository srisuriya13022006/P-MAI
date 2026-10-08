from datetime import datetime
import logging

from app.database.models import Memory
from app.database.repositories.memories import MemoryRepository
from app.memory.conflict import ConflictDetector
from app.memory.extractor import MemoryExtractor
from app.memory.llm_extractor import LLMExtractionProvider, LLMMemoryExtractor
from app.memory.profile import MemoryProfileBuilder
from app.memory.retrieval import MemoryRetriever
from app.memory.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderFactory,
    LocalHashEmbeddingProvider,
    cosine_similarity,
)
from app.schemas.memory import MemoryCandidate

logger = logging.getLogger(__name__)

# Cosine-similarity threshold above which two memories are considered duplicates.
_DUPLICATE_SIMILARITY_THRESHOLD = 0.97


class MemoryService:
    """Coordinates extraction, filtering, persistence, retrieval, and updates."""

    def __init__(
        self,
        repository: MemoryRepository,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str | None = None,
        llm: LLMExtractionProvider | None = None,
        profile_builder: MemoryProfileBuilder | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = (
            embedding_provider
            or EmbeddingProviderFactory.create(embedding_model)
            or LocalHashEmbeddingProvider()
        )
        llm_extractor = LLMMemoryExtractor(llm) if llm is not None else None
        self.extractor = MemoryExtractor(llm_extractor=llm_extractor)
        self.retriever = MemoryRetriever(self.embedding_provider)
        self.conflict_detector = ConflictDetector()
        self.profile_builder = profile_builder or MemoryProfileBuilder()

    def build_user_profile(self, user_id: str, now: datetime | None = None) -> str:
        """Build a formatted user profile context string for new conversations."""
        logger.debug("Building user memory profile for user '%s'", user_id)
        memories = self.repository.list_for_user(user_id)
        return self.profile_builder.build(memories, now=now)

    def get_user_profile_memories(
        self,
        user_id: str,
        now: datetime | None = None,
    ) -> list[Memory]:
        """Return raw selected profile memories (useful for context deduplication)."""
        memories = self.repository.list_for_user(user_id)
        return self.profile_builder.select_memories(memories, now=now)

    def remember_from_message(self, user_id: str, message: str) -> int:
        saved = 0
        logger.debug("Memory extraction started for user '%s'.", user_id)
        candidates = self.extractor.extract(message)
        if not candidates:
            return 0
        # Pre-fetch once to avoid N+1 queries inside the loop.
        existing_memories = self.repository.list_for_user(user_id)
        for candidate in candidates:
            active_existing = [m for m in existing_memories if getattr(m, "is_active", True)]

            # 1. Exact-content duplicate check
            if any(m.content == candidate.content for m in active_existing):
                logger.debug("Exact duplicate skipped: %r", candidate.content)
                continue

            # 2. Semantic near-duplicate check using embeddings
            candidate_embedding = self.embedding_provider.embed(candidate.content)
            if self._is_near_duplicate(candidate_embedding, candidate.memory_type, active_existing):
                logger.debug("Near-duplicate memory skipped: %r", candidate.content)
                continue

            # 3. Conflict detection against active existing memories
            conflicting_memory = self.conflict_detector.find_conflict(
                candidate,
                active_existing,
            )

            if conflicting_memory:
                new_memory = self.repository.create(
                    user_id=user_id,
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    importance=candidate.importance,
                    confidence=candidate.confidence,
                    embedding=candidate_embedding,
                    is_active=True,
                )
                self.repository.mark_superseded(conflicting_memory, new_memory.id)
                existing_memories.append(new_memory)
                logger.debug(
                    "Conflict resolved: %r (id=%s) superseded by %r (id=%s)",
                    conflicting_memory.content,
                    conflicting_memory.id,
                    new_memory.content,
                    new_memory.id,
                )
                saved += 1
                continue

            # 4. New memory: persist as active
            new_memory = self.repository.create(
                user_id=user_id,
                content=candidate.content,
                memory_type=candidate.memory_type,
                importance=candidate.importance,
                confidence=candidate.confidence,
                embedding=candidate_embedding,
                is_active=True,
            )
            existing_memories.append(new_memory)
            logger.debug("Memory persisted: %r", candidate.content)
            saved += 1
        logger.debug("Memory extraction complete — %d memory/ies saved.", saved)
        return saved
    
    def remember(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: int = 2,
    ):
        existing = self.repository.find_by_content(user_id, content)
        if existing and getattr(existing, "is_active", True):
            return existing

        candidate = MemoryCandidate(
            content=content,
            memory_type=memory_type,
            importance=importance,
            confidence=1.0,
        )
        existing_memories = self.repository.list_for_user(user_id)
        candidate_embedding = self.embedding_provider.embed(content)

        active_existing = [m for m in existing_memories if getattr(m, "is_active", True)]
        conflicting_memory = self.conflict_detector.find_conflict(
            candidate,
            active_existing,
        )

        new_memory = self.repository.create(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            confidence=1.0,
            embedding=candidate_embedding,
            is_active=True,
        )
        if conflicting_memory:
            self.repository.mark_superseded(conflicting_memory, new_memory.id)
            logger.debug(
                "Explicit remember conflict: %r superseded by %r",
                conflicting_memory.content,
                new_memory.content,
            )

        return new_memory

    def search(self, user_id: str, query: str):
        memories = self.retriever.retrieve(query, self.repository.list_for_user(user_id))
        now = datetime.utcnow()
        for memory in memories:
            memory.last_accessed_at = now
        return memories

    def update(self, user_id: str, memory_id: int, content: str):
        memory = self.repository.get(user_id, memory_id)
        if memory:
            memory.content = content
            memory.embedding = self.embedding_provider.embed(content)
            memory.updated_at = datetime.utcnow()
        return memory

    def relevant_context(self, user_id: str, query: str) -> list[str]:
        memories = self.retriever.retrieve(
            query,
            self.repository.list_for_user(user_id),
        )
        now = datetime.utcnow()
        for memory in memories:
            memory.last_accessed_at = now
        return [memory.content for memory in memories]

    def forget(
        self,
        user_id: str,
        memory_id: int | None = None,
        query: str | None = None,
    ) -> bool:
        """Forget one memory, scoped to its owning user."""
        if memory_id is None and query:
            memory = self.repository.find_matching(user_id, query)
            memory_id = memory.id if memory else None
        if memory_id is None:
            return False
        return self.repository.delete(user_id, memory_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_near_duplicate(
        self,
        candidate_embedding: list[float],
        memory_type: str,
        existing_memories: list | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Return True if a semantically near-identical memory already exists.

        Checks only memories of the same type to avoid cross-type false positives
        (e.g. a "goal" should not suppress a "preference" with similar wording).
        """
        if existing_memories is None:
            if user_id is None:
                return False
            existing_memories = self.repository.list_for_user(user_id)
        for memory in existing_memories:
            if not getattr(memory, "is_active", True):
                continue
            if memory.memory_type != memory_type:
                continue
            if not memory.embedding:
                continue
            similarity = cosine_similarity(candidate_embedding, memory.embedding)
            if similarity >= _DUPLICATE_SIMILARITY_THRESHOLD:
                return True
        return False