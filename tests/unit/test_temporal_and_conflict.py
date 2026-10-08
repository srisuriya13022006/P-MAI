"""Comprehensive test suite for Phase 1 & 2 Memory Enhancements:
- Temporal / time-aware memory scoring & decay
- Conflict detection, resolution, and versioning
- Duplicate vs conflict distinction
- Hybrid retrieval with freshness ranking
"""

from datetime import datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.connection import Base
from app.database.models import Memory
from app.database.repositories.memories import MemoryRepository
from app.memory.conflict import ConflictDetector
from app.memory.embeddings import LocalHashEmbeddingProvider
from app.memory.retrieval import MemoryRetriever
from app.memory.service import MemoryService
from app.memory.temporal import calculate_temporal_score, is_expired
from app.schemas.memory import MemoryCandidate


def _make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine


# ===========================================================================
# 1. TEMPORAL SCORING & DECAY TESTS
# ===========================================================================


class TestTemporalScoring:
    def test_very_recent_memory_has_high_freshness(self):
        """A memory created now has score 1.0."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        mem = Memory(
            user_id="u1",
            content="User is learning PyTorch",
            memory_type="goal",
            importance=2,
            created_at=now,
            updated_at=now,
        )
        score = calculate_temporal_score(mem, now=now)
        assert score == 1.0

    def test_older_goal_receives_gradual_decay(self):
        """A 60-day old goal decays according to ~30-day half-life."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        old_time = now - timedelta(days=60)
        mem = Memory(
            user_id="u1",
            content="User is learning TensorFlow",
            memory_type="goal",
            importance=2,  # half life = 30 days
            created_at=old_time,
            updated_at=old_time,
        )
        score = calculate_temporal_score(mem, now=now)
        # 60 days / 30 day half-life = 2 half-lives => 0.5^2 = 0.25
        assert 0.20 <= score <= 0.30

    def test_high_importance_permanent_fact_protected_from_aggressive_decay(self):
        """Permanent facts and high-importance memories never drop below the 0.80 floor."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        very_old = now - timedelta(days=1000)  # ~3 years old
        name_fact = Memory(
            user_id="u1",
            content="User's name is Suriya",
            memory_type="fact",
            importance=3,
            created_at=very_old,
            updated_at=very_old,
        )
        score = calculate_temporal_score(name_fact, now=now)
        assert score >= 0.80

    def test_expired_memory_returns_zero_score_and_is_expired(self):
        """A memory with expires_at in the past is flagged expired and gets score 0.0."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        expired_mem = Memory(
            user_id="u1",
            content="User is visiting London this week",
            memory_type="fact",
            importance=1,
            created_at=now - timedelta(days=10),
            updated_at=now - timedelta(days=10),
            expires_at=now - timedelta(days=1),
        )
        assert is_expired(expired_mem, now=now) is True
        assert calculate_temporal_score(expired_mem, now=now) == 0.0

    def test_future_expiration_memory_is_not_expired(self):
        """A memory with expires_at in the future is not expired."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        active_mem = Memory(
            user_id="u1",
            content="User is on vacation until next week",
            memory_type="fact",
            importance=1,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=7),
        )
        assert is_expired(active_mem, now=now) is False
        assert calculate_temporal_score(active_mem, now=now) == 1.0

    def test_missing_timestamp_fails_safely(self):
        """Memory missing both created_at and updated_at fails safely with score 1.0."""
        mem = Memory(
            user_id="u1",
            content="Some memory",
            memory_type="fact",
            importance=2,
            created_at=None,
            updated_at=None,
        )
        score = calculate_temporal_score(mem)
        assert score == 1.0

    def test_retrieval_ranking_prefers_fresher_memories(self):
        """When two memories have similar lexical/semantic relevance, the newer one ranks higher."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        provider = LocalHashEmbeddingProvider()
        retriever = MemoryRetriever(provider)

        old_mem = Memory(
            id=1,
            user_id="u1",
            content="User is learning TensorFlow",
            memory_type="goal",
            importance=2,
            created_at=now - timedelta(days=90),
            updated_at=now - timedelta(days=90),
            is_active=True,
        )
        new_mem = Memory(
            id=2,
            user_id="u1",
            content="User is learning PyTorch",
            memory_type="goal",
            importance=2,
            created_at=now - timedelta(hours=1),
            updated_at=now - timedelta(hours=1),
            is_active=True,
        )

        results = retriever.retrieve("What framework is user learning?", [old_mem, new_mem], now=now)
        assert len(results) == 2
        # Newer memory should rank first
        assert results[0].content == "User is learning PyTorch"
        assert results[1].content == "User is learning TensorFlow"

    def test_retrieval_filters_out_inactive_and_expired(self):
        """Inactive or expired memories are excluded from retrieval results."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        provider = LocalHashEmbeddingProvider()
        retriever = MemoryRetriever(provider)

        active = Memory(
            id=1, user_id="u1", content="User prefers Python", memory_type="preference",
            importance=2, created_at=now, updated_at=now, is_active=True,
        )
        superseded = Memory(
            id=2, user_id="u1", content="User prefers JavaScript", memory_type="preference",
            importance=2, created_at=now - timedelta(days=10), updated_at=now - timedelta(days=10),
            is_active=False, superseded_by=1,
        )
        expired = Memory(
            id=3, user_id="u1", content="User prefers Ruby for this week", memory_type="preference",
            importance=2, created_at=now - timedelta(days=5), updated_at=now - timedelta(days=5),
            expires_at=now - timedelta(days=1), is_active=True,
        )

        results = retriever.retrieve("What programming language does user prefer?", [active, superseded, expired], now=now)
        contents = [m.content for m in results]
        assert "User prefers Python" in contents
        assert "User prefers JavaScript" not in contents
        assert "User prefers Ruby for this week" not in contents


# ===========================================================================
# 2. DUPLICATE VS CONFLICT DISTINCTION TESTS
# ===========================================================================


class TestDuplicateVsConflict:
    def test_exact_duplicate_suppression(self):
        """Exact duplicate message stores only once."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            assert svc.remember_from_message("u1", "I prefer C++ for DSA.") == 1
            db.commit()
            assert svc.remember_from_message("u1", "I prefer C++ for DSA.") == 0
            db.commit()

            memories = MemoryRepository(db).list_for_user("u1")
            assert len(memories) == 1

    def test_semantic_near_duplicate_suppression(self):
        """Near-identical wording of same type is suppressed as a duplicate."""
        engine = _make_db()
        provider = LocalHashEmbeddingProvider()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db), embedding_provider=provider)
            svc.remember("u1", "User prefers C++ for DSA", memory_type="preference")
            db.commit()

            embedding = provider.embed("User prefers C++ for DSA")
            existing = MemoryRepository(db).list_for_user("u1")
            # Near duplicate detection
            assert svc._is_near_duplicate(embedding, "preference", existing) is True

    def test_different_type_similar_wording_not_suppressed(self):
        """Similar wording across different memory types are both retained."""
        engine = _make_db()
        provider = LocalHashEmbeddingProvider()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db), embedding_provider=provider)
            svc.remember("u1", "User prefers Python", memory_type="preference")
            db.commit()

            embedding = provider.embed("User prefers Python")
            existing = MemoryRepository(db).list_for_user("u1")
            # Different type ('goal') should not be suppressed by 'preference'
            assert svc._is_near_duplicate(embedding, "goal", existing) is False


# ===========================================================================
# 3. CONFLICT DETECTION & VERSIONING TESTS
# ===========================================================================


class TestConflictDetectionAndResolution:
    def test_preference_conflict_python_to_javascript(self):
        """Changing programming language preference supersedes old and activates new."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            # 1. User says they prefer Python
            assert svc.remember_from_message("u1", "I prefer Python.") == 1
            db.commit()

            active_memories = repo.list_for_user("u1")
            assert len(active_memories) == 1
            assert active_memories[0].content == "User prefers Python"
            old_id = active_memories[0].id

            # 2. User later says they prefer JavaScript
            assert svc.remember_from_message("u1", "I prefer JavaScript.") == 1
            db.commit()

            # Active list must contain only the new preference
            active_after = repo.list_for_user("u1")
            assert len(active_after) == 1
            assert active_after[0].content == "User prefers JavaScript"
            new_id = active_after[0].id

            # All records (including historical inactive)
            all_memories = repo.list_for_user("u1", include_inactive=True)
            assert len(all_memories) == 2

            old_mem = next(m for m in all_memories if m.id == old_id)
            assert old_mem.is_active is False
            assert old_mem.superseded_by == new_id

            # Retrieval returns ONLY the active preference
            context = svc.relevant_context("u1", "What programming language do I prefer?")
            assert context == ["User prefers JavaScript"]

    def test_goal_conflict_old_goal_to_new_goal(self):
        """Updating a goal in the same domain supersedes the old goal."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            # Initial goal
            assert svc.remember_from_message("u1", "I am learning TensorFlow.") == 1
            db.commit()
            old_mem = repo.list_for_user("u1")[0]

            # New goal in same domain
            assert svc.remember_from_message("u1", "I am learning PyTorch.") == 1
            db.commit()

            active = repo.list_for_user("u1")
            assert len(active) == 1
            assert active[0].content == "User is learning PyTorch"

            # Check superseded link
            all_mems = repo.list_for_user("u1", include_inactive=True)
            superseded = next(m for m in all_mems if m.id == old_mem.id)
            assert superseded.is_active is False
            assert superseded.superseded_by == active[0].id

    def test_project_state_conflict(self):
        """Project state updates for the same project entity supersede previous state."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            svc.remember("u1", "User is building P-MAI in Python", memory_type="project")
            db.commit()
            old_id = repo.list_for_user("u1")[0].id

            # New project update for P-MAI
            svc.remember("u1", "User is building P-MAI in Rust", memory_type="project")
            db.commit()

            active = repo.list_for_user("u1")
            assert len(active) == 1
            assert active[0].content == "User is building P-MAI in Rust"

            all_mems = repo.list_for_user("u1", include_inactive=True)
            old_proj = next(m for m in all_mems if m.id == old_id)
            assert old_proj.is_active is False
            assert old_proj.superseded_by == active[0].id

    def test_fact_conflict_name_change(self):
        """Updating user name conflicts and supersedes old name fact."""
        candidate = MemoryCandidate(content="User's name is Alex", memory_type="fact", importance=3)
        existing = Memory(id=1, user_id="u1", content="User's name is Suriya", memory_type="fact", importance=3, is_active=True)

        conflict = ConflictDetector.find_conflict(candidate, [existing])
        assert conflict is not None
        assert conflict.id == 1

    def test_non_conflicting_preferences_coexist(self):
        """Preferences in different domains (e.g. programming language vs theme) do not conflict."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            svc.remember("u1", "User prefers Python", memory_type="preference")
            db.commit()

            svc.remember("u1", "User likes dark mode", memory_type="preference")
            db.commit()

            active = repo.list_for_user("u1")
            assert len(active) == 2
            contents = [m.content for m in active]
            assert "User prefers Python" in contents
            assert "User likes dark mode" in contents

    def test_non_conflicting_projects_coexist(self):
        """Two different projects do not conflict."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            svc.remember("u1", "User is building P-MAI", memory_type="project")
            db.commit()

            svc.remember("u1", "User is building an eCommerce app", memory_type="project")
            db.commit()

            active = repo.list_for_user("u1")
            assert len(active) == 2

    def test_different_memory_types_do_not_conflict(self):
        """A preference and a goal do not conflict even if mentioned in the same context."""
        candidate = MemoryCandidate(content="User is currently learning Kubernetes", memory_type="goal", importance=2)
        existing = Memory(id=1, user_id="u1", content="User prefers Python", memory_type="preference", importance=2, is_active=True)

        conflict = ConflictDetector.find_conflict(candidate, [existing])
        assert conflict is None

    def test_last_accessed_at_touched_on_search(self):
        """Retrieving memories touches their last_accessed_at timestamp."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))
            repo = MemoryRepository(db)

            svc.remember("u1", "User prefers Python", memory_type="preference")
            db.commit()

            mem = repo.list_for_user("u1")[0]
            assert mem.last_accessed_at is None

            svc.relevant_context("u1", "What language do I prefer?")
            db.commit()

            updated_mem = repo.list_for_user("u1")[0]
            assert updated_mem.last_accessed_at is not None
