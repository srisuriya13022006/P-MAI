"""Tests for policy routing fixes, memory update ordering, and concise response quality.

Verifies:
1. "I'm using Python now." must NOT become web_search.
2. "I am currently learning FastAPI." must NOT become web_search.
3. "What is the latest AI news?" MUST remain web_search.
4. "What are the recent updates to Python?" MUST remain web_search.
5. Preference update supersedes old C++ preference and makes Python active.
6. Relevant context during/after update does not include the superseded C++ preference.
7. System prompt explicitly instructs concise responses for simple updates and prohibits unsolicited tutorials.
8. System prompt permits detailed answers for complex/explicit explanation requests.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agent.policy import apply_policy
from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.memory.embeddings import LocalHashEmbeddingProvider
from app.prompts.conversation import MAI_SYSTEM_PROMPT
from app.schemas.agent import AgentDecision
from app.schemas.memory import MemoryCandidate
from app.services.chat_service import ChatService


@pytest.fixture(autouse=True)
def fast_embeddings(monkeypatch):
    monkeypatch.setattr(
        "app.memory.embeddings.EmbeddingProviderFactory.create",
        lambda *args, **kwargs: LocalHashEmbeddingProvider(),
    )


def _base_decision() -> AgentDecision:
    return AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )


def test_personal_statement_with_now_does_not_become_web_search():
    """1. 'I'm using Python now.' must NOT be hijacked to web_search."""
    decision = _base_decision()
    routed = apply_policy("I'm using Python now.", decision)
    assert routed.route != "tool" or "web_search" not in routed.tools


def test_personal_statement_with_currently_does_not_become_web_search():
    """2. 'I am currently learning FastAPI.' must NOT become web_search."""
    for message in (
        "I am currently learning FastAPI.",
        "I prefer Python.",
        "I no longer prefer C++.",
        "My current goal is to master DSA.",
        "I am currently building P-MAI.",
    ):
        decision = _base_decision()
        routed = apply_policy(message, decision)
        assert routed.route != "tool" or "web_search" not in routed.tools, f"Failed for '{message}'"


def test_latest_ai_news_routes_to_web_search():
    """3. 'What is the latest AI news?' MUST remain web_search."""
    decision = _base_decision()
    routed = apply_policy("What is the latest AI news?", decision)
    assert routed.route == "tool"
    assert "web_search" in routed.tools


def test_recent_updates_routes_to_web_search():
    """4. 'What are the recent updates to Python?' MUST remain web_search."""
    decision = _base_decision()
    routed = apply_policy("What are the recent updates to Python?", decision)
    assert routed.route == "tool"
    assert "web_search" in routed.tools


def test_preference_update_supersedes_old_and_activates_new():
    """5. Preference update: old C++ memory is superseded, Python memory is active."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        repo = MemoryRepository(db)
        # Seed existing active memory
        old_mem = repo.create(
            user_id="user-1",
            content="User prefers C++ for DSA",
            memory_type="preference",
            importance=3,
            is_active=True,
        )
        db.commit()

        # Mock LLM extractor that extracts the new preference
        class MockLLM:
            def generate(self, messages, system_prompt=None):
                if system_prompt and "memory extraction" in system_prompt.lower():
                    return '{"memories": [{"type": "preference", "content": "User prefers Python for DSA", "confidence": 0.98}]}'
                return "Got it. I'll treat Python as your current DSA preference."

        service = ChatService(db, llm=MockLLM())
        service.orchestrator.router.analyze = lambda msg: _base_decision()

        service.chat(
            conversation_id="conv-1",
            user_id="user-1",
            user_message="I no longer prefer C++ for DSA. I'm using Python now.",
        )
        service.chat(
            conversation_id="conv-1",
            user_id="user-1",
            user_message="Yes.",
        )

        all_memories = repo.list_for_user("user-1", include_inactive=True)
        old_memories = [m for m in all_memories if "C++" in m.content]
        new_memories = [m for m in all_memories if "Python" in m.content]

        assert len(old_memories) == 1
        assert old_memories[0].is_active is False
        assert old_memories[0].superseded_by is not None

        assert len(new_memories) == 1
        assert new_memories[0].is_active is True


def test_relevant_context_after_update_excludes_superseded_memory():
    """6. Relevant context during and after update does not include the superseded C++ preference."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        repo = MemoryRepository(db)
        repo.create(
            user_id="user-1",
            content="User prefers C++ for DSA",
            memory_type="preference",
            importance=3,
            is_active=True,
        )
        db.commit()

        class MockLLM:
            def generate(self, messages, system_prompt=None):
                if system_prompt and "memory extraction" in system_prompt.lower():
                    return '{"memories": [{"type": "preference", "content": "User prefers Python for DSA", "confidence": 0.98}]}'
                return "Got it. I'll remember Python for DSA."

        service = ChatService(db, llm=MockLLM())
        service.orchestrator.router.analyze = lambda msg: _base_decision()

        captured_memory_contexts = []
        original_handle = service.orchestrator.handle

        def mock_handle(*args, **kwargs):
            captured_memory_contexts.append(kwargs.get("memory_context") or [])
            return "Got it. I'll remember Python for DSA."

        service.orchestrator.handle = mock_handle

        service.chat(
            conversation_id="conv-1",
            user_id="user-1",
            user_message="I no longer prefer C++ for DSA. I'm using Python now.",
        )

        assert len(captured_memory_contexts) == 1
        context = captured_memory_contexts[0]
        # Superseded C++ preference must NOT be present in memory context
        assert not any("C++" in mem for mem in context)


def test_conversation_prompt_specifies_concise_guidance_for_updates():
    """7. System prompt instructs concise responses for updates and prohibits unsolicited tutorials."""
    assert "briefly in 1–2 conversational sentences" in MAI_SYSTEM_PROMPT
    assert "Do not provide unsolicited tutorials" in MAI_SYSTEM_PROMPT
    assert "acknowledge the update naturally" in MAI_SYSTEM_PROMPT


def test_conversation_prompt_permits_detailed_answers_for_complex_requests():
    """8. System prompt permits detailed answers when explanation is explicitly requested."""
    assert "detailed, thorough answers" in MAI_SYSTEM_PROMPT


def test_what_do_you_know_about_me_uses_profile_response_path():
    """'What do you know about me?' must NOT route to remember_memory or literal search_memory."""
    decision = _base_decision()
    routed = apply_policy("What do you know about me?", decision)
    assert routed.route == "memory"
    assert "remember_memory" not in routed.tools
    assert "search_memory" not in routed.tools
    assert routed.tools == []


def test_what_do_you_remember_about_me_uses_profile_response_path():
    """'What do you remember about me?' must use profile/memory route."""
    decision = _base_decision()
    routed = apply_policy("What do you remember about me?", decision)
    assert routed.route == "memory"
    assert "remember_memory" not in routed.tools
    assert "search_memory" not in routed.tools
    assert routed.tools == []


def test_what_is_my_name_is_read_not_remember_memory():
    """'What is my name?' is a READ query and must NOT use remember_memory."""
    # Even if router mistakenly proposed remember_memory
    decision = _base_decision()
    decision.tools = ["remember_memory"]
    decision.route = "tool"
    routed = apply_policy("What is my name?", decision)
    assert routed.route == "memory"
    assert "remember_memory" not in routed.tools
    assert routed.tools == []


def test_specific_memory_queries_use_retrieval_flow():
    """Specific queries like 'What do I prefer for DSA?' and 'What project am I building?' work correctly."""
    for query in ("What do I prefer for DSA?", "What project am I building?"):
        decision = _base_decision()
        routed = apply_policy(query, decision)
        assert routed.route != "tool" or "web_search" not in routed.tools
        assert "remember_memory" not in routed.tools


def test_remember_write_command_preserved():
    """'Remember that I prefer Python.' preserves remember_memory write tool."""
    decision = _base_decision()
    routed = apply_policy("Remember that I prefer Python.", decision)
    assert routed.route == "tool"
    assert routed.tools == ["remember_memory"]
    assert routed.tool_arguments.get("content") == "I prefer Python"


def test_broad_profile_query_synthesizes_profile_in_chat_service():
    """Verifies ChatService injects user profile for 'What do you know about me?' even in existing conversations."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        repo = MemoryRepository(db)
        repo.create(user_id="user-1", content="User's name is Suriya", memory_type="fact", importance=3)
        repo.create(user_id="user-1", content="User is building P-MAI", memory_type="project", importance=3)
        repo.create(user_id="user-1", content="User currently uses Python for DSA", memory_type="preference", importance=3)
        db.commit()

        captured_system_prompts = []

        class MockLLM:
            def generate(self, messages, system_prompt=None):
                if system_prompt:
                    captured_system_prompts.append(system_prompt)
                return "You're Suriya, building P-MAI and using Python for DSA."

        service = ChatService(db, llm=MockLLM())
        # Simulate turn 2 (conversation already has messages)
        service.chat(conversation_id="conv-1", user_id="user-1", user_message="Hello")
        captured_system_prompts.clear()

        response = service.chat(
            conversation_id="conv-1",
            user_id="user-1",
            user_message="What do you know about me?",
        )

        assert "Suriya" in response
        assert len(captured_system_prompts) >= 1
        prompt = captured_system_prompts[-1]
        assert "USER PROFILE:" in prompt
        assert "Suriya" in prompt
        assert "P-MAI" in prompt
        assert "Python" in prompt
