"""Tests for Phase 3: Cross-Session Memory Continuity & User Profile Context.

Verifies:
- Profile generation and deterministic ranking
- Importance and temporal weighting
- Empty profile safety
- User isolation (User A vs User B)
- Conflict state handling (superseded memories excluded)
- Conversation startup injection (new conversation vs ongoing conversation)
- Duplicate context prevention between profile and per-message retrieval
"""

from datetime import datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.connection import Base
from app.database.models import Memory
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.memories import MemoryRepository
from app.memory.profile import MemoryProfileBuilder
from app.memory.service import MemoryService
from app.services.chat_service import ChatService


def _make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine


class TestUserProfileBuilder:
    def test_basic_profile_generation(self):
        """User profile formats high-value memories with canonical phrasing."""
        memories = [
            Memory(id=1, user_id="u1", content="User's name is Suriya", memory_type="fact", importance=3, is_active=True),
            Memory(id=2, user_id="u1", content="User is building P-MAI", memory_type="project", importance=3, is_active=True),
            Memory(id=3, user_id="u1", content="User prefers C++ for DSA", memory_type="preference", importance=2, is_active=True),
            Memory(id=4, user_id="u1", content="User is learning FastAPI", memory_type="goal", importance=2, is_active=True),
        ]
        builder = MemoryProfileBuilder()
        profile = builder.build(memories)

        assert "USER PROFILE:" in profile
        assert "- User's name is Suriya." in profile
        assert "- User is building P-MAI." in profile
        assert "- User prefers C++ for DSA." in profile
        assert "- User is learning FastAPI." in profile

    def test_empty_profile_returns_empty_string(self):
        """When a user has no memories, profile builder safely returns empty string."""
        builder = MemoryProfileBuilder()
        profile = builder.build([])
        assert profile == ""

    def test_importance_selection_under_limit(self):
        """When memories exceed max_memories, high importance memories are chosen over low ones."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        memories = [
            Memory(id=1, user_id="u1", content="User ate pizza for lunch", memory_type="fact", importance=1, created_at=now, updated_at=now, is_active=True),
            Memory(id=2, user_id="u1", content="User likes blue", memory_type="preference", importance=1, created_at=now, updated_at=now, is_active=True),
            Memory(id=3, user_id="u1", content="User's name is Suriya", memory_type="fact", importance=3, created_at=now, updated_at=now, is_active=True),
            Memory(id=4, user_id="u1", content="User is building P-MAI", memory_type="project", importance=3, created_at=now, updated_at=now, is_active=True),
        ]
        builder = MemoryProfileBuilder(max_memories=2)
        selected = builder.select_memories(memories, now=now)

        assert len(selected) == 2
        selected_ids = {m.id for m in selected}
        assert 3 in selected_ids  # Name (importance 3)
        assert 4 in selected_ids  # P-MAI project (importance 3)

    def test_expired_and_inactive_memories_excluded(self):
        """Inactive (superseded) and expired memories never appear in the user profile."""
        now = datetime(2026, 10, 3, 12, 0, 0)
        memories = [
            Memory(id=1, user_id="u1", content="User prefers Python", memory_type="preference", importance=2, is_active=True, created_at=now, updated_at=now),
            Memory(id=2, user_id="u1", content="User prefers JavaScript", memory_type="preference", importance=2, is_active=False, superseded_by=1, created_at=now, updated_at=now),
            Memory(id=3, user_id="u1", content="User is traveling this week", memory_type="fact", importance=1, is_active=True, created_at=now - timedelta(days=5), updated_at=now - timedelta(days=5), expires_at=now - timedelta(days=1)),
        ]
        builder = MemoryProfileBuilder()
        selected = builder.select_memories(memories, now=now)

        contents = [m.content for m in selected]
        assert "User prefers Python" in contents
        assert "User prefers JavaScript" not in contents
        assert "User is traveling this week" not in contents

    def test_character_budget_limit(self):
        """Profile builder strictly respects character limits."""
        memories = [
            Memory(id=1, user_id="u1", content="A" * 60, memory_type="fact", importance=3, is_active=True),
            Memory(id=2, user_id="u1", content="B" * 60, memory_type="fact", importance=3, is_active=True),
            Memory(id=3, user_id="u1", content="C" * 60, memory_type="fact", importance=3, is_active=True),
        ]
        builder = MemoryProfileBuilder(max_memories=10, max_characters=140)
        selected = builder.select_memories(memories)
        assert len(selected) == 2


class TestUserProfileIsolation:
    def test_cross_user_isolation(self):
        """User A's profile never contains memories belonging to User B."""
        engine = _make_db()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db))

            svc.remember("user-A", "User's name is Alice", memory_type="fact", importance=3)
            svc.remember("user-A", "User prefers Python", memory_type="preference", importance=2)

            svc.remember("user-B", "User's name is Bob", memory_type="fact", importance=3)
            svc.remember("user-B", "User prefers Go", memory_type="preference", importance=2)
            db.commit()

            profile_a = svc.build_user_profile("user-A")
            profile_b = svc.build_user_profile("user-B")

            assert "Alice" in profile_a
            assert "Python" in profile_a
            assert "Bob" not in profile_a
            assert "Go" not in profile_a

            assert "Bob" in profile_b
            assert "Go" in profile_b
            assert "Alice" not in profile_b
            assert "Python" not in profile_b


class TestChatServiceProfileIntegration:
    def test_new_conversation_injects_profile(self):
        """Starting a new conversation injects the user profile into the initial turn."""
        engine = _make_db()
        with Session(engine) as db:
            service = ChatService(db)
            service.memory_service.remember("user-1", "User prefers C++", memory_type="preference", importance=3)
            db.commit()

            captured_profile = []

            # Spy on orchestrator.handle
            original_handle = service.orchestrator.handle

            def mock_handle(*args, **kwargs):
                captured_profile.append(kwargs.get("user_profile"))
                return "Mock assistant response"

            service.orchestrator.handle = mock_handle

            # Turn 1: Brand new conversation
            service.chat(
                conversation_id="conv-1",
                user_id="user-1",
                user_message="Hello!",
            )

            assert len(captured_profile) == 1
            assert captured_profile[0] is not None
            assert "USER PROFILE:" in captured_profile[0]
            assert "User prefers C++" in captured_profile[0]

    def test_existing_conversation_does_not_reinject_profile(self):
        """Turn 2 in an existing conversation does not re-inject the profile."""
        engine = _make_db()
        with Session(engine) as db:
            service = ChatService(db)
            service.memory_service.remember("user-1", "User prefers C++", memory_type="preference", importance=3)
            db.commit()

            captured_profiles = []

            def mock_handle(*args, **kwargs):
                captured_profiles.append(kwargs.get("user_profile"))
                return "Mock assistant response"

            service.orchestrator.handle = mock_handle

            # Turn 1: New conversation
            service.chat(
                conversation_id="conv-1",
                user_id="user-1",
                user_message="Hello!",
            )
            # Turn 2: Ongoing conversation
            service.chat(
                conversation_id="conv-1",
                user_id="user-1",
                user_message="Can you help me with a coding problem?",
            )

            assert len(captured_profiles) == 2
            # Turn 1 has profile
            assert captured_profiles[0] is not None
            assert "USER PROFILE:" in captured_profiles[0]
            # Turn 2 does NOT re-inject profile
            assert captured_profiles[1] is None

    def test_duplicate_context_prevention(self):
        """Memories already in the profile are not duplicated in the per-message memory_context."""
        engine = _make_db()
        with Session(engine) as db:
            service = ChatService(db)
            service.memory_service.remember("user-1", "User prefers C++", memory_type="preference", importance=3)
            service.memory_service.remember("user-1", "User likes dark mode", memory_type="preference", importance=2)
            db.commit()

            captured_memory_context = []

            def mock_handle(*args, **kwargs):
                captured_memory_context.append(kwargs.get("memory_context"))
                return "Mock response"

            service.orchestrator.handle = mock_handle

            # User asks explicitly about their C++ preference
            service.chat(
                conversation_id="conv-1",
                user_id="user-1",
                user_message="What programming language do I prefer?",
            )

            assert len(captured_memory_context) == 1
            # "User prefers C++" is already in the USER PROFILE, so it must NOT be duplicated in memory_context
            profile_memories = service.memory_service.get_user_profile_memories("user-1")
            profile_texts = [m.content.strip().lower() for m in profile_memories]
            assert "user prefers c++" in profile_texts

            context = captured_memory_context[0] or []
            assert "User prefers C++" not in context
