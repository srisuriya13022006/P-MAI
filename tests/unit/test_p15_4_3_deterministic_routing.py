"""
P15.4.3 Regression Tests:
Restoring Deterministic Fast Paths for Self-Identity and Datetime Routing.
Asserts:
- 'What is your name?', 'Who are you?', 'Are you MAI?' use deterministic local response with 0 LLM calls and 0 memory calls.
- 'What time is it?' uses DateTimeTool with 0 LLM calls and configured Asia/Kolkata default timezone (no invented location).
- Explicit timezone queries ('What time is it in Tokyo?') use the requested timezone without LLM calls.
- RequestAnalyzer.analyze makes zero LLM calls for these queries.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.orchestrator import MAIOrchestrator
from app.agent.policy import apply_policy, is_pure_datetime_query, is_self_identity_query
from app.agent.router import RequestAnalyzer
from app.core.config import settings
from app.database.connection import Base
from app.schemas.agent import AgentDecision
from app.services.chat_service import ChatService
from app.tools.datetime.tool import DateTimeTool


@pytest.fixture
def fixed_clock():
    # Tuesday, 2026-10-06 12:00:00 UTC
    fixed_utc = datetime(2026, 10, 6, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    return lambda tz: fixed_utc.astimezone(tz)


@pytest.mark.parametrize(
    "query",
    [
        "What is your name?",
        "Who are you?",
        "Are you MAI?",
    ],
)
def test_self_identity_fast_path_zero_llm_zero_memory(query: str):
    """Assert zero LLM calls and zero memory calls for self-identity queries in Orchestrator and ChatService."""
    assert is_self_identity_query(query) is True

    # 1. Router analyze makes zero LLM calls
    mock_router_llm = MagicMock()
    analyzer = RequestAnalyzer(llm=mock_router_llm)
    decision = analyzer.analyze(query)
    assert decision.route == "local"
    assert decision.intent == "identify_self"
    assert decision.tools == []
    mock_router_llm.generate.assert_not_called()

    # 2. Orchestrator handle makes zero LLM and zero memory calls
    mock_memory_service = MagicMock()
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(memory_service=mock_memory_service, llm=mock_llm)
    orchestrator.router.llm = mock_router_llm

    resp = orchestrator.handle(user_message=query, conversation_messages=[])
    assert "MAI" in resp
    mock_llm.generate.assert_not_called()
    mock_router_llm.generate.assert_not_called()
    mock_memory_service.relevant_context.assert_not_called()

    # 3. ChatService chat makes zero LLM and zero memory calls
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()

    service = ChatService(db=db, llm=mock_llm)
    with patch.object(service.memory_service, "relevant_context") as mock_rel, \
         patch.object(service.memory_service, "build_user_profile") as mock_prof:
        chat_resp = service.chat("conv-test-id", "user-test-id", query)
        assert "MAI" in chat_resp
        mock_llm.generate.assert_not_called()
        mock_rel.assert_not_called()
        mock_prof.assert_not_called()


def test_ordinary_datetime_zero_llm_calls_and_kolkata_timezone(fixed_clock):
    """
    Assert 'What time is it?' uses local DateTimeTool,
    makes zero LLM calls, zero memory calls, and uses configured Asia/Kolkata timezone without invented location.
    """
    query = "What time is it?"
    assert is_pure_datetime_query(query) is True

    # 1. Router analyze makes zero LLM calls
    mock_router_llm = MagicMock()
    analyzer = RequestAnalyzer(llm=mock_router_llm)
    decision = analyzer.analyze(query)
    assert decision.route == "tool"
    assert decision.intent == "datetime"
    assert decision.tools == ["datetime"]
    assert decision.tool_arguments.get("location") is None
    mock_router_llm.generate.assert_not_called()

    # 2. Orchestrator handle makes zero LLM calls and returns configured default timezone
    mock_llm = MagicMock()
    mock_cloud_llm = MagicMock()
    mock_memory_service = MagicMock()
    orchestrator = MAIOrchestrator(memory_service=mock_memory_service, llm=mock_llm)
    orchestrator.cloud_llm = mock_cloud_llm
    orchestrator.router.llm = mock_router_llm
    orchestrator.tool_registry._tools["datetime"] = DateTimeTool(clock=fixed_clock)

    resp = orchestrator.handle(user_message=query, conversation_messages=[])

    # In fixed_clock (12:00 UTC), Asia/Kolkata is UTC+5:30 -> 17:30:00
    assert "17:30:00" in resp
    assert "Asia/Kolkata" in resp
    assert "New York" not in resp  # No invented location!
    assert "America/New_York" not in resp
    mock_llm.generate.assert_not_called()
    mock_cloud_llm.generate.assert_not_called()
    mock_router_llm.generate.assert_not_called()
    mock_memory_service.relevant_context.assert_not_called()


def test_explicit_timezone_datetime_lookup(fixed_clock):
    """
    Assert 'What time is it in Tokyo?' resolves Tokyo timezone (Asia/Tokyo)
    with zero LLM calls and correct localized time.
    """
    query = "What time is it in Tokyo?"
    assert is_pure_datetime_query(query) is True

    mock_router_llm = MagicMock()
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(llm=mock_llm)
    orchestrator.router.llm = mock_router_llm
    orchestrator.tool_registry._tools["datetime"] = DateTimeTool(clock=fixed_clock)

    resp = orchestrator.handle(user_message=query, conversation_messages=[])

    # In fixed_clock (12:00 UTC), Asia/Tokyo is UTC+9:00 -> 21:00:00
    assert "21:00:00" in resp
    assert "Asia/Tokyo" in resp
    assert "Tokyo" in resp
    mock_llm.generate.assert_not_called()
    mock_router_llm.generate.assert_not_called()


def test_hallucinated_location_is_purged_on_generic_query():
    """
    If existing arguments contained a hallucinated location ('New York'),
    resolve_datetime_arguments on 'What time is it?' must purge it.
    """
    from app.agent.tool_argument_resolver import resolve_datetime_arguments

    res = resolve_datetime_arguments("What time is it?", {"location": "New York", "timezone": "America/New_York"})
    assert res.get("location") is None
    assert res.get("timezone") is None
    assert res.get("query_type") == "time"


def test_web_search_with_today_keyword_is_not_intercepted_by_datetime():
    """'Search the web for today's AI news.' must route to web_search, not datetime."""
    query = "Search the web for today's AI news."
    assert is_pure_datetime_query(query) is False

    decision = AgentDecision(
        intent="general",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )
    routed = apply_policy(query, decision)
    assert routed.route == "tool"
    assert routed.tools == ["web_search"]
