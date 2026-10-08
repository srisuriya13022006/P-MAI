"""Integration tests for ChatService and POST /chat LLM provider wiring.

Verifies:
A. ChatService wiring with mock LLM provider and default canonical provider
B. POST /chat with "I'm building P-MAI." invokes LLM extractor and persists project memory
C. POST /chat with "I'm currently learning FastAPI." invokes LLM extractor and persists goal memory
D. LLM failure falls back cleanly to RegexMemoryExtractor
E. Real /chat dependency path supplies canonical provider instance
"""

import json
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.database.session import get_db
from app.llm.ollama_provider import OllamaProvider, get_ollama_provider
from app.main import app
from app.memory.embeddings import LocalHashEmbeddingProvider
from app.memory.llm_extractor import LLMMemoryExtractor
from app.schemas.agent import AgentDecision
from app.services.chat_service import ChatService


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Ensure fast, deterministic test environment without external model timeouts."""
    monkeypatch.setattr(
        "app.agent.router.RequestAnalyzer.analyze",
        lambda self, msg: AgentDecision(
            intent="local",
            route="local",
            needs_clarification=False,
            tools=[],
            tool_arguments={},
            reason="test local route",
        ),
    )
    monkeypatch.setattr(
        "app.memory.embeddings.EmbeddingProviderFactory.create",
        lambda *args, **kwargs: LocalHashEmbeddingProvider(),
    )


@pytest.fixture
def test_db_setup():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    yield TestingSessionLocal, override_get_db
    Base.metadata.drop_all(engine)


class FakeMockLLM:
    """Mock LLM provider that simulates both extraction and orchestrator responses."""

    def __init__(self, extraction_payload: dict | None = None, raise_error: bool = False):
        self.extraction_payload = extraction_payload or {"memories": []}
        self.raise_error = raise_error
        self.calls = []

    def generate(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        if self.raise_error:
            raise RuntimeError("Simulated LLM connection failure")

        # Distinguish extraction prompt from conversational generation
        if system_prompt and "memory extraction" in system_prompt.lower():
            return json.dumps(self.extraction_payload)
        return "I understand. I have updated my context."


def test_chat_service_wires_llm_provider(test_db_setup):
    """Test A: ChatService with a mock LLM provider wires LLMMemoryExtractor."""
    TestingSessionLocal, _ = test_db_setup
    with TestingSessionLocal() as db:
        mock_llm = FakeMockLLM()
        service = ChatService(db, llm=mock_llm)

        # 1. MemoryService receives the provider
        assert service.memory_service.extractor._llm is not None
        assert isinstance(service.memory_service.extractor._llm, LLMMemoryExtractor)
        assert service.memory_service.extractor._llm.llm is mock_llm

        # 2. MAIOrchestrator shares the same provider
        assert service.orchestrator.llm is mock_llm


def test_chat_service_canonical_provider_by_default(test_db_setup):
    """ChatService without explicit llm uses the canonical get_ollama_provider instance."""
    TestingSessionLocal, _ = test_db_setup
    with TestingSessionLocal() as db:
        service = ChatService(db)

        assert service.memory_service.extractor._llm is not None
        assert isinstance(service.memory_service.extractor._llm, LLMMemoryExtractor)
        assert service.memory_service.extractor._llm.llm is service.orchestrator.llm
        assert isinstance(service.orchestrator.llm, OllamaProvider)


def test_post_chat_persists_project_memory_via_llm_extractor(test_db_setup):
    """Test B: POST /chat message 'I'm building P-MAI.' invokes LLM extractor and persists project memory."""
    TestingSessionLocal, override_get_db = test_db_setup
    mock_llm = FakeMockLLM(
        extraction_payload={
            "memories": [
                {
                    "type": "project",
                    "content": "User is building P-MAI",
                    "confidence": 0.98,
                }
            ]
        }
    )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama_provider] = lambda: mock_llm

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "user_id": "user-test-b",
                "conversation_id": "conv-test-b",
                "message": "I'm building P-MAI.",
            },
        )
        assert response.status_code == 200

        # Confirm candidate memory write
        client.post(
            "/chat",
            json={
                "user_id": "user-test-b",
                "conversation_id": "conv-test-b",
                "message": "Yes.",
            },
        )

        # Verify LLM was invoked for extraction
        extraction_calls = [
            call for call in mock_llm.calls
            if call["system_prompt"] and "memory extraction" in call["system_prompt"].lower()
        ]
        assert len(extraction_calls) >= 1

        # Verify project memory is persisted in DB
        with TestingSessionLocal() as db:
            repo = MemoryRepository(db)
            memories = repo.list_for_user("user-test-b")
            assert len(memories) == 1
            assert memories[0].memory_type == "project"
            assert memories[0].content == "User is building P-MAI"
            assert memories[0].importance == 3
    finally:
        app.dependency_overrides.clear()


def test_post_chat_persists_goal_memory_via_llm_extractor(test_db_setup):
    """Test C: POST /chat message 'I'm currently learning FastAPI.' invokes LLM extractor and persists goal memory."""
    TestingSessionLocal, override_get_db = test_db_setup
    mock_llm = FakeMockLLM(
        extraction_payload={
            "memories": [
                {
                    "type": "goal",
                    "content": "User is learning FastAPI",
                    "confidence": 0.95,
                }
            ]
        }
    )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama_provider] = lambda: mock_llm

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "user_id": "user-test-c",
                "conversation_id": "conv-test-c",
                "message": "I'm currently learning FastAPI.",
            },
        )
        assert response.status_code == 200

        # Confirm candidate memory write
        client.post(
            "/chat",
            json={
                "user_id": "user-test-c",
                "conversation_id": "conv-test-c",
                "message": "Yes.",
            },
        )

        # Verify LLM was invoked for extraction
        extraction_calls = [
            call for call in mock_llm.calls
            if call["system_prompt"] and "memory extraction" in call["system_prompt"].lower()
        ]
        assert len(extraction_calls) >= 1

        # Verify goal memory is persisted in DB
        with TestingSessionLocal() as db:
            repo = MemoryRepository(db)
            memories = repo.list_for_user("user-test-c")
            assert len(memories) == 1
            assert memories[0].memory_type == "goal"
            assert memories[0].content == "User is learning FastAPI"
    finally:
        app.dependency_overrides.clear()


def test_post_chat_llm_failure_falls_back_to_regex(test_db_setup):
    """Test D: When LLM fails, RegexMemoryExtractor fallback still works."""
    TestingSessionLocal, override_get_db = test_db_setup
    failing_llm = FakeMockLLM(raise_error=True)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama_provider] = lambda: failing_llm

    try:
        client = TestClient(app)
        # "My name is Suriya." matches the deterministic regex pattern
        response = client.post(
            "/chat",
            json={
                "user_id": "user-test-d",
                "conversation_id": "conv-test-d",
                "message": "My name is Suriya.",
            },
        )
        assert response.status_code == 200

        # Confirm candidate memory write
        client.post(
            "/chat",
            json={
                "user_id": "user-test-d",
                "conversation_id": "conv-test-d",
                "message": "Yes.",
            },
        )

        # Memory was extracted and persisted via RegexMemoryExtractor fallback
        with TestingSessionLocal() as db:
            repo = MemoryRepository(db)
            memories = repo.list_for_user("user-test-d")
            assert len(memories) == 1
            assert memories[0].memory_type == "fact"
            assert memories[0].content == "User's name is Suriya"
    finally:
        app.dependency_overrides.clear()


def test_real_chat_dependency_path_supplies_provider():
    """Verify that the real /chat route dependency path actually supplies an OllamaProvider."""
    canonical_provider = get_ollama_provider()
    assert isinstance(canonical_provider, OllamaProvider)

    # Check the endpoint dependency signature
    from app.api.routes_chat import chat
    import inspect
    sig = inspect.signature(chat)
    assert "llm" in sig.parameters
    param = sig.parameters["llm"]
    assert param.default.dependency == get_ollama_provider
