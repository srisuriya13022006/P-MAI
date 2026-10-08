"""
P15.1 — Self-Identity Routing Regression & Fast-Path Tests.
Validates explicit deterministic separation between:
- MAI Self-Identity queries (route='local', tools=[], intent='identify_self')
- User Memory queries (route='memory', appropriate memory behavior)
Tests route decisions directly for: intent, route, needs_clarification, tools, reason.
Also tests ChatService and Voice E2E integration without memory leakage.
"""
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.agent.policy import apply_policy, is_self_identity_query
from app.agent.orchestrator import MAIOrchestrator
from app.database.connection import Base
from app.database.session import get_db
from app.llm.ollama_provider import get_ollama_provider
from app.main import app
from app.schemas.agent import AgentDecision
from app.services.chat_service import ChatService


def _base_decision() -> AgentDecision:
    return AgentDecision(
        intent="general",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="Base test decision",
    )


# ---------------------------------------------------------------------------
# 1. Direct Policy Routing Tests for MAI Self-Identity (Cases 1-9)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "What is your name?",
        "Who are you?",
        "What are you?",
        "Are you an AI?",
        "Tell me about yourself.",
        "What can you do?",
        "Hi, what's your name?",
        "Hey MAI, who are you?",
        "Bye, what is your name?",
    ],
)
def test_self_identity_routing_decisions(query: str):
    """
    All self-identity queries must route to 'local' with intent='identify_self',
    tools=[], needs_clarification=False.
    """
    # Even if router initially proposed 'memory' or another route:
    decision = _base_decision()
    decision.route = "memory"
    decision.intent = "identify_self"

    routed = apply_policy(query, decision)

    assert routed.route == "local", f"Failed for query: {query}"
    assert routed.intent == "identify_self", f"Failed for query: {query}"
    assert routed.tools == [], f"Failed for query: {query}"
    assert routed.needs_clarification is False, f"Failed for query: {query}"
    assert "self-identity" in routed.reason.lower() or "local" in routed.reason.lower()


# ---------------------------------------------------------------------------
# 2. Direct Policy Routing Tests for User Memory (Cases 10-13)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "What is my name?",
        "What do you remember about me?",
        "Do you remember my project?",
        "What is my preferred language?",
        "Do you remember my name?",
    ],
)
def test_user_memory_routing_decisions(query: str):
    """
    User memory queries must preserve route='memory' and NOT route to 'local'.
    """
    decision = _base_decision()
    decision.route = "memory"

    routed = apply_policy(query, decision)

    assert routed.route == "memory", f"Failed for query: {query}"
    assert routed.intent != "identify_self", f"Failed for query: {query}"
    assert routed.tools == [], f"Failed for query: {query}"
    assert routed.needs_clarification is False, f"Failed for query: {query}"


# ---------------------------------------------------------------------------
# 3. Helper Function Semantics
# ---------------------------------------------------------------------------
def test_is_self_identity_query_helper():
    # Self-identity true
    assert is_self_identity_query("What is your name?") is True
    assert is_self_identity_query("what's your name") is True
    assert is_self_identity_query("whats your name") is True
    assert is_self_identity_query("who are you") is True
    assert is_self_identity_query("who r u") is True
    assert is_self_identity_query("what are you") is True
    assert is_self_identity_query("are you an ai") is True
    assert is_self_identity_query("tell me about yourself") is True
    assert is_self_identity_query("what can you do") is True
    assert is_self_identity_query("Hi, what's your name?") is True
    assert is_self_identity_query("Bye, what is your name?") is True
    assert is_self_identity_query("Hello! Are you an AI?") is True

    # User memory false
    assert is_self_identity_query("What is my name?") is False
    assert is_self_identity_query("What do you know about me?") is False
    assert is_self_identity_query("Tell me about myself") is False
    assert is_self_identity_query("Do you remember my name?") is False
    assert is_self_identity_query("What is my preferred language?") is False
    assert is_self_identity_query("What do you remember about me?") is False
    assert is_self_identity_query("Do you remember my project?") is False


# ---------------------------------------------------------------------------
# 4. Orchestrator Fast-Path Execution
# ---------------------------------------------------------------------------
def test_orchestrator_fast_path_self_identity():
    mock_memory_service = MagicMock()
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(memory_service=mock_memory_service, llm=mock_llm)

    response = orchestrator.handle(
        user_message="What is your name?",
        conversation_messages=[{"role": "user", "content": "What is your name?"}],
        memory_context=[],
        user_id="u-1",
        conversation_id="c-1",
    )

    assert "I'm MAI" in response
    # Fast path must NOT call LLM generate or memory queries
    mock_llm.generate.assert_not_called()
    mock_memory_service.relevant_context.assert_not_called()


def test_orchestrator_capabilities_self_identity():
    mock_memory_service = MagicMock()
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(memory_service=mock_memory_service, llm=mock_llm)

    response = orchestrator.handle(
        user_message="What can you do?",
        conversation_messages=[{"role": "user", "content": "What can you do?"}],
        memory_context=[],
        user_id="u-1",
        conversation_id="c-1",
    )

    assert "I'm MAI" in response
    assert "help" in response.lower()
    mock_llm.generate.assert_not_called()


def test_orchestrator_greeting_farewell_combined():
    mock_memory_service = MagicMock()
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(memory_service=mock_memory_service, llm=mock_llm)

    # Combined with greeting
    res_hi = orchestrator.handle(
        user_message="Hi, what's your name?",
        conversation_messages=[],
        memory_context=[],
        user_id="u-1",
        conversation_id="c-1",
    )
    assert "I'm MAI" in res_hi

    # Combined with farewell
    res_bye = orchestrator.handle(
        user_message="Bye, what is your name?",
        conversation_messages=[],
        memory_context=[],
        user_id="u-1",
        conversation_id="c-1",
    )
    assert "I'm MAI" in res_bye
    mock_llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# 5. ChatService Integration (Zero Memory Leakage)
# ---------------------------------------------------------------------------
def test_chat_service_self_identity_no_memory_calls():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()

    mock_llm = MagicMock()
    service = ChatService(db=db, llm=mock_llm)

    with patch.object(service.memory_service, "remember_from_message") as mock_remember, \
         patch.object(service.memory_service, "relevant_context") as mock_relevant:

        answer = service.chat("conv-id-1", "user-id-1", "What is your name?")
        assert "I'm MAI" in answer
        # Verify memory extraction was NOT performed for self-identity query
        mock_remember.assert_not_called()
        mock_relevant.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Voice E2E Integration Regression Test
# ---------------------------------------------------------------------------
def test_voice_realtime_self_identity():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()

    app.dependency_overrides[get_db] = lambda: db
    mock_llm = MagicMock()
    app.dependency_overrides[get_ollama_provider] = lambda: mock_llm

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/voice/realtime?conversation_id=self-id-conv") as ws:
            # Start speech turn
            ws.send_json({"type": "AUDIO_START"})
            ws.receive_json()  # SPEECH_START

            # Mock transcription returning 'What is your name?'
            with patch("app.voice.stt.streaming.MockStreamingSTTProvider.finalize_stream") as mock_stt:
                from app.voice.schemas import STTResult
                mock_stt.return_value = STTResult(transcript="What is your name?", provider="mock")

                ws.send_json({"type": "AUDIO_END"})
                msg_speech_end = ws.receive_json()  # SPEECH_END
                msg_transcript = ws.receive_json()  # TRANSCRIPT_FINAL
                assert msg_transcript["transcript"] == "What is your name?"

                # Next events: RESPONSE_START -> RESPONSE_CHUNK -> RESPONSE_END
                received_types = []
                final_text = ""
                while True:
                    evt = ws.receive_json()
                    received_types.append(evt["type"])
                    if evt["type"] == "RESPONSE_END":
                        final_text = evt.get("text_chunk", "")
                        break

                assert "RESPONSE_START" in received_types
                assert "I'm MAI" in final_text

    app.dependency_overrides.clear()
