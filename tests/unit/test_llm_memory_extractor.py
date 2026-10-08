"""Tests for the LLM-based memory extractor and the hybrid MemoryExtractor.

Covers:
- Positive extraction cases (fact, preference, goal, project)
- Indirect-language cases the regex extractor would miss
- Negative cases (questions, transient statements, hypotheticals)
- Robustness cases (LLM failure, invalid JSON, empty response, bad schema)
- Duplicate detection in MemoryService (exact and near-duplicate)
- Regression: all existing MemoryService / regex behaviour unchanged
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.memory.embeddings import LocalHashEmbeddingProvider
from app.memory.extractor import MemoryExtractor, RegexMemoryExtractor
from app.memory.llm_extractor import LLMMemoryExtractor
from app.memory.service import MemoryService


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    """LLM fake that returns a fixed JSON string."""

    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, messages, system_prompt=None) -> str:  # noqa: ANN001
        return self._response


class _BrokenLLM:
    """LLM fake that always raises."""

    def generate(self, messages, system_prompt=None) -> str:  # noqa: ANN001
        raise RuntimeError("Simulated LLM failure")


class _TimeoutLLM:
    """LLM fake that simulates a timeout (raises TimeoutError)."""

    def generate(self, messages, system_prompt=None) -> str:  # noqa: ANN001
        raise TimeoutError("Simulated timeout")


def _llm_extractor(response: str) -> LLMMemoryExtractor:
    return LLMMemoryExtractor(_FakeLLM(response))


def _service(db: Session, llm=None) -> MemoryService:
    return MemoryService(
        MemoryRepository(db),
        embedding_provider=LocalHashEmbeddingProvider(),
        llm=llm,
    )


def _make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine


# ===========================================================================
# LLMMemoryExtractor — positive cases
# ===========================================================================


class TestLLMMemoryExtractorPositive:
    def test_extracts_fact(self):
        response = '{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.99}]}'
        candidates = _llm_extractor(response).extract("My name is Suriya.")
        assert len(candidates) == 1
        assert candidates[0].memory_type == "fact"
        assert "Suriya" in candidates[0].content
        assert candidates[0].importance == 3  # confidence >= 0.95

    def test_extracts_preference(self):
        response = '{"memories": [{"type": "preference", "content": "User prefers C++ for DSA", "confidence": 0.97}]}'
        candidates = _llm_extractor(response).extract("I prefer C++ for DSA.")
        assert len(candidates) == 1
        assert candidates[0].memory_type == "preference"
        assert candidates[0].importance == 3

    def test_extracts_goal(self):
        response = '{"memories": [{"type": "goal", "content": "User is preparing for AI engineering placements", "confidence": 0.96}]}'
        candidates = _llm_extractor(response).extract("I'm preparing for AI engineering placements.")
        assert len(candidates) == 1
        assert candidates[0].memory_type == "goal"

    def test_extracts_project(self):
        response = '{"memories": [{"type": "project", "content": "User is building a personal agentic assistant called P-MAI", "confidence": 0.98}]}'
        candidates = _llm_extractor(response).extract("I'm building a personal agentic assistant called P-MAI.")
        assert len(candidates) == 1
        assert candidates[0].memory_type == "project"

    def test_extracts_multiple_memories(self):
        response = """{
            "memories": [
                {"type": "fact", "content": "User's name is Suriya", "confidence": 0.99},
                {"type": "goal", "content": "User is learning FastAPI", "confidence": 0.95}
            ]
        }"""
        candidates = _llm_extractor(response).extract("My name is Suriya and I'm learning FastAPI.")
        assert len(candidates) == 2

    def test_indirect_name_phrasing(self):
        response = '{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.98}]}'
        candidates = _llm_extractor(response).extract("Call me Suriya.")
        assert len(candidates) == 1
        assert "Suriya" in candidates[0].content

    def test_indirect_dislike_phrasing(self):
        response = '{"memories": [{"type": "preference", "content": "User does not like working in the morning", "confidence": 0.93}]}'
        candidates = _llm_extractor(response).extract("I don't like working in the morning.")
        assert len(candidates) == 1
        assert candidates[0].memory_type == "preference"
        assert candidates[0].importance == 2  # 0.80 <= conf < 0.95

    def test_indirect_coding_habit(self):
        response = '{"memories": [{"type": "preference", "content": "User usually codes in C++", "confidence": 0.94}]}'
        candidates = _llm_extractor(response).extract("I usually code in C++.")
        assert len(candidates) == 1

    def test_future_goal_phrasing(self):
        response = '{"memories": [{"type": "goal", "content": "User wants to start learning Kubernetes", "confidence": 0.90}]}'
        candidates = _llm_extractor(response).extract("Next month I want to start learning Kubernetes.")
        assert len(candidates) == 1

    def test_importance_low_for_low_confidence(self):
        response = '{"memories": [{"type": "preference", "content": "User might prefer tea", "confidence": 0.70}]}'
        candidates = _llm_extractor(response).extract("I might prefer tea.")
        assert candidates[0].importance == 1

    def test_importance_medium_for_mid_confidence(self):
        response = '{"memories": [{"type": "preference", "content": "User prefers tea", "confidence": 0.85}]}'
        candidates = _llm_extractor(response).extract("I prefer tea.")
        assert candidates[0].importance == 2


# ===========================================================================
# LLMMemoryExtractor — negative cases
# ===========================================================================


class TestLLMMemoryExtractorNegative:
    def test_question_returns_empty(self):
        response = '{"memories": []}'
        candidates = _llm_extractor(response).extract("What is FastAPI?")
        assert candidates == []

    def test_transient_statement_returns_empty(self):
        response = '{"memories": []}'
        candidates = _llm_extractor(response).extract("I'm hungry right now.")
        assert candidates == []

    def test_hypothetical_returns_empty(self):
        response = '{"memories": []}'
        candidates = _llm_extractor(response).extract("If I learn Rust, what should I use for web?")
        assert candidates == []

    def test_general_knowledge_returns_empty(self):
        response = '{"memories": []}'
        candidates = _llm_extractor(response).extract("Python is a programming language.")
        assert candidates == []

    def test_one_time_request_returns_empty(self):
        response = '{"memories": []}'
        candidates = _llm_extractor(response).extract("Explain recursion to me.")
        assert candidates == []


# ===========================================================================
# LLMMemoryExtractor — robustness cases
# ===========================================================================


class TestLLMMemoryExtractorRobustness:
    def test_llm_failure_returns_empty(self):
        extractor = LLMMemoryExtractor(_BrokenLLM())
        result = extractor.extract("My name is Suriya.")
        assert result == []

    def test_llm_timeout_returns_empty(self):
        extractor = LLMMemoryExtractor(_TimeoutLLM())
        result = extractor.extract("My name is Suriya.")
        assert result == []

    def test_invalid_json_returns_empty(self):
        result = _llm_extractor("not json at all").extract("My name is Suriya.")
        assert result == []

    def test_empty_response_returns_empty(self):
        result = _llm_extractor("").extract("My name is Suriya.")
        assert result == []

    def test_json_missing_memories_key_returns_empty(self):
        result = _llm_extractor('{"data": []}').extract("My name is Suriya.")
        assert result == []

    def test_unsupported_memory_type_is_skipped(self):
        response = '{"memories": [{"type": "habit", "content": "User sleeps at 11pm", "confidence": 0.9}]}'
        result = _llm_extractor(response).extract("I sleep at 11pm.")
        assert result == []

    def test_empty_content_is_skipped(self):
        response = '{"memories": [{"type": "fact", "content": "  ", "confidence": 0.99}]}'
        result = _llm_extractor(response).extract("My name is Suriya.")
        assert result == []

    def test_confidence_out_of_range_is_rejected(self):
        response = '{"memories": [{"type": "fact", "content": "User likes tea", "confidence": 1.5}]}'
        result = _llm_extractor(response).extract("I like tea.")
        assert result == []

    def test_markdown_fenced_json_is_parsed(self):
        """LLM wraps output in ```json fences — must still work."""
        response = '```json\n{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.99}]}\n```'
        result = _llm_extractor(response).extract("My name is Suriya.")
        assert len(result) == 1

    def test_missing_confidence_defaults_gracefully(self):
        response = '{"memories": [{"type": "fact", "content": "User\'s name is Suriya"}]}'
        result = _llm_extractor(response).extract("My name is Suriya.")
        assert len(result) == 1
        assert result[0].confidence == 1.0

    def test_partial_valid_and_invalid_items(self):
        """Only valid items should be kept; invalid ones silently skipped."""
        response = """{
            "memories": [
                {"type": "fact", "content": "User's name is Suriya", "confidence": 0.99},
                {"type": "unknown_type", "content": "something", "confidence": 0.9}
            ]
        }"""
        result = _llm_extractor(response).extract("My name is Suriya.")
        assert len(result) == 1
        assert "Suriya" in result[0].content


# ===========================================================================
# Hybrid MemoryExtractor — fallback behaviour
# ===========================================================================


class TestHybridMemoryExtractor:
    def test_uses_llm_when_available_and_succeeds(self):
        llm_ext = LLMMemoryExtractor(
            _FakeLLM('{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.99}]}')
        )
        extractor = MemoryExtractor(llm_extractor=llm_ext)
        result = extractor.extract("Call me Suriya.")
        assert len(result) == 1
        assert "Suriya" in result[0].content

    def test_falls_back_to_regex_when_llm_fails(self):
        llm_ext = LLMMemoryExtractor(_BrokenLLM())
        extractor = MemoryExtractor(llm_extractor=llm_ext)
        # Regex can handle "I prefer C++"
        result = extractor.extract("I prefer C++.")
        assert len(result) == 1
        assert result[0].memory_type == "preference"

    def test_falls_back_to_regex_when_llm_returns_empty(self):
        llm_ext = LLMMemoryExtractor(_FakeLLM('{"memories": []}'))
        extractor = MemoryExtractor(llm_extractor=llm_ext)
        result = extractor.extract("I prefer Python.")
        assert len(result) == 1

    def test_no_llm_uses_only_regex(self):
        extractor = MemoryExtractor()  # no LLM — pure regex
        result = extractor.extract("I prefer Python.")
        assert len(result) == 1
        assert result[0].memory_type == "preference"

    def test_no_llm_regex_misses_indirect_phrasing(self):
        """Confirm regex cannot handle 'Call me Suriya' — LLM is needed."""
        regex_ext = RegexMemoryExtractor()
        result = regex_ext.extract("Call me Suriya.")
        assert result == []

    def test_llm_result_preferred_over_regex(self):
        """When LLM gives a richer result, regex should not also run."""
        llm_ext = LLMMemoryExtractor(
            _FakeLLM('{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.99}]}')
        )
        extractor = MemoryExtractor(llm_extractor=llm_ext)
        # Message also matches "I prefer" regex pattern — but LLM wins
        result = extractor.extract("My name is Suriya and I prefer Python.")
        # LLM returned exactly 1 memory, so exactly 1 is expected
        assert len(result) == 1
        assert "Suriya" in result[0].content


# ===========================================================================
# MemoryService — duplicate prevention
# ===========================================================================


class TestMemoryServiceDuplicatePrevention:
    def test_exact_duplicate_not_stored_twice(self):
        engine = _make_db()
        with Session(engine) as db:
            svc = _service(db)
            svc.remember("user-1", "User prefers Python", memory_type="preference")
            db.commit()
            # Same exact content — should not duplicate
            svc.remember("user-1", "User prefers Python", memory_type="preference")
            db.commit()
            memories = MemoryRepository(db).list_for_user("user-1")
            assert len(memories) == 1

    def test_near_duplicate_not_stored_when_similarity_high(self):
        """Two near-identical strings should not create two DB rows."""
        engine = _make_db()
        provider = LocalHashEmbeddingProvider()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db), embedding_provider=provider)
            # Store via remember() (explicit)
            svc.remember("user-1", "User prefers C++ for DSA", memory_type="preference")
            db.commit()
            # Manually test _is_near_duplicate with the same content's embedding
            embedding = provider.embed("User prefers C++ for DSA")
            existing = MemoryRepository(db).list_for_user("user-1")
            assert svc._is_near_duplicate(embedding, "preference", existing) is True

    def test_different_type_not_treated_as_duplicate(self):
        """Same wording with different type should not be suppressed."""
        engine = _make_db()
        provider = LocalHashEmbeddingProvider()
        with Session(engine) as db:
            svc = MemoryService(MemoryRepository(db), embedding_provider=provider)
            svc.remember("user-1", "User prefers Python", memory_type="preference")
            db.commit()
            embedding = provider.embed("User prefers Python")
            existing = MemoryRepository(db).list_for_user("user-1")
            # Goal type should NOT be flagged as duplicate of preference
            assert svc._is_near_duplicate(embedding, "goal", existing) is False


# ===========================================================================
# MemoryService integration with LLM extractor
# ===========================================================================


class TestMemoryServiceLLMIntegration:
    def test_service_wires_llm_extractor_when_provided(self):
        engine = _make_db()
        fake_llm = _FakeLLM(
            '{"memories": [{"type": "fact", "content": "User\'s name is Suriya", "confidence": 0.99}]}'
        )
        with Session(engine) as db:
            svc = _service(db, llm=fake_llm)
            saved = svc.remember_from_message("user-1", "Call me Suriya.")
            db.commit()
            assert saved == 1
            memories = MemoryRepository(db).list_for_user("user-1")
            assert any("Suriya" in m.content for m in memories)

    def test_service_falls_back_to_regex_when_llm_breaks(self):
        engine = _make_db()
        with Session(engine) as db:
            svc = _service(db, llm=_BrokenLLM())
            saved = svc.remember_from_message("user-1", "I prefer Python.")
            db.commit()
            assert saved == 1

    def test_service_without_llm_works_as_before(self):
        engine = _make_db()
        with Session(engine) as db:
            svc = _service(db)  # no LLM — pure regex
            saved = svc.remember_from_message("user-1", "I prefer C++.")
            db.commit()
            assert saved == 1

    def test_llm_non_memory_message_saves_nothing(self):
        engine = _make_db()
        fake_llm = _FakeLLM('{"memories": []}')
        with Session(engine) as db:
            svc = _service(db, llm=fake_llm)
            saved = svc.remember_from_message("user-1", "What is 2 + 2?")
            db.commit()
            assert saved == 0


# ===========================================================================
# Regression: original MemoryService behaviour unchanged
# ===========================================================================


class TestMemoryServiceRegression:
    """All tests from the original test_memory.py must continue to pass."""

    def test_memory_is_filtered_retrieved_and_updated(self):
        engine = _make_db()
        with Session(engine) as db:
            service = MemoryService(MemoryRepository(db))

            assert service.remember_from_message("user-1", "I prefer C++.") == 1
            assert service.remember_from_message("user-1", "I ate dosa for lunch.") == 0
            db.commit()

            context = service.relevant_context(
                "user-1",
                "What programming language do I prefer?",
            )
            assert context == ["User prefers C++"]

            assert service.remember_from_message("user-1", "I prefer Python.") == 1
            db.commit()
            memories = MemoryRepository(db).list_for_user("user-1")
            assert [m.content for m in memories] == ["User prefers Python"]

            assert service.forget("user-1", memories[0].id) is True
            db.commit()
            assert service.relevant_context("user-1", "What language do I prefer?") == []
