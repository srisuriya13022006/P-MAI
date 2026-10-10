"""
Tests for P15.4.4 — Groq TPM Optimization and Fast Conversational Routing.
Verifies zero unnecessary LLM calls for greetings, acknowledgments, farewells, identity,
datetime, and arithmetic, per-turn LLM accounting, token capping, and 429 fallback handling.
"""
import pytest
from app.agent.policy import (
    is_greeting_query,
    is_acknowledgment_query,
    is_farewell_query,
    is_pure_calculator_query,
    is_self_identity_query,
    is_pure_datetime_query,
    apply_policy,
)
from app.agent.orchestrator import MAIOrchestrator
from app.agent.router import RequestAnalyzer
from app.schemas.agent import AgentDecision
from app.llm.resilient_provider import ResilientLLMProvider


class CountingMockLLM:
    """Mock LLM provider that tracks call counts, stage arguments, and returns fixed responses."""
    def __init__(self, name: str = "mock_llm", response: str = "mock answer"):
        self.provider_name = name
        self.model = "mock-model"
        self.calls = []
        self.response = response
        self.last_usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        self.last_status = "200_OK"
        self.last_retry_after = None

    def generate(self, messages, system_prompt=None, stage="conversation", max_tokens=None, **kwargs):
        self.calls.append({
            "stage": stage,
            "max_tokens": max_tokens,
            "messages": messages,
            "system_prompt": system_prompt,
        })
        return self.response


class CountingFailLLM:
    """Mock LLM that simulates HTTP 429 RateLimitError."""
    def __init__(self, retry_after: float = 10.875):
        self.provider_name = "groq"
        self.model = "openai/gpt-oss-20b"
        self.calls = []
        self.last_usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        self.last_status = "429_RATE_LIMIT"
        self.last_retry_after = retry_after

    def generate(self, messages, system_prompt=None, stage="conversation", max_tokens=None, **kwargs):
        self.calls.append({"stage": stage, "max_tokens": max_tokens})
        raise RuntimeError(f"Rate limit reached on TPM. Please try again in {self.last_retry_after}s.")


def test_fast_conversational_policy_classifiers():
    """Verify regex classifiers for greetings, acknowledgments, farewells, and arithmetic."""
    # Greetings
    assert is_greeting_query("Hello")
    assert is_greeting_query("hello.")
    assert is_greeting_query("Hey")
    assert is_greeting_query("Good morning")
    assert is_greeting_query("Hi MAI")
    assert not is_greeting_query("Hello what is your name?")
    assert not is_greeting_query("Hello search the web for news")

    # Acknowledgments
    assert is_acknowledgment_query("All okay")
    assert is_acknowledgment_query("all okay.")
    assert is_acknowledgment_query("okay")
    assert is_acknowledgment_query("Sounds good")
    assert is_acknowledgment_query("Got it")
    assert is_acknowledgment_query("Thanks!")
    assert not is_acknowledgment_query("All okay, calculate 5+5")

    # Farewells
    assert is_farewell_query("Goodbye")
    assert is_farewell_query("bye")
    assert is_farewell_query("See you later")
    assert is_farewell_query("Bye MAI")
    assert not is_farewell_query("Goodbye, what time is it?")

    # Pure calculator
    assert is_pure_calculator_query("What is 25 times 47?")
    assert is_pure_calculator_query("calculate 15 + 28")
    assert is_pure_calculator_query("25 * 47")
    assert is_pure_calculator_query("100 divided by 4")
    assert is_pure_calculator_query("What is 7 * 8?")
    assert not is_pure_calculator_query("What is the date 10 days after today?")
    assert not is_pure_calculator_query("Search for news about python 3.12")
    assert not is_pure_calculator_query("compare python 3.12 with 3.13")


def test_router_deterministic_fast_paths_make_zero_llm_calls():
    """Verify RequestAnalyzer uses deterministic fast paths without invoking LLM."""
    mock_llm = CountingMockLLM()
    router = RequestAnalyzer(llm=mock_llm)

    # 1. Greeting
    dec_greet = router.analyze("Hello")
    assert dec_greet.route == "local"
    assert dec_greet.intent == "greet"
    assert len(mock_llm.calls) == 0

    # 2. Acknowledgment
    dec_ack = router.analyze("All okay")
    assert dec_ack.route == "local"
    assert dec_ack.intent == "acknowledge"
    assert len(mock_llm.calls) == 0

    # 3. Farewell
    dec_farewell = router.analyze("Goodbye")
    assert dec_farewell.route == "local"
    assert dec_farewell.intent == "farewell"
    assert len(mock_llm.calls) == 0

    # 4. Identity
    dec_id = router.analyze("What is your name?")
    assert dec_id.route == "local"
    assert dec_id.intent == "identify_self"
    assert len(mock_llm.calls) == 0

    # 5. Datetime
    dec_dt = router.analyze("What time is it?")
    assert dec_dt.route == "tool"
    assert dec_dt.intent == "datetime"
    assert len(mock_llm.calls) == 0

    # 6. Calculator
    dec_calc = router.analyze("What is 25 times 47?")
    assert dec_calc.route == "tool"
    assert dec_calc.intent == "arithmetic"
    assert dec_calc.tools == ["calculator"]
    assert len(mock_llm.calls) == 0


def test_calculator_query_makes_zero_llm_calls_in_orchestrator():
    """Verify 'What is 25 times 47?' executes calculator tool with zero LLM calls."""
    mock_llm = CountingMockLLM()
    orchestrator = MAIOrchestrator(llm=mock_llm)
    orchestrator.cloud_llm = None

    response = orchestrator.handle("What is 25 times 47?", [])
    assert "1175" in response
    assert len(mock_llm.calls) == 0, "Calculator query must make 0 LLM calls"


def test_goodbye_makes_zero_llm_calls_in_orchestrator():
    """Verify 'Goodbye' uses deterministic farewell fast path with zero LLM calls."""
    mock_llm = CountingMockLLM()
    orchestrator = MAIOrchestrator(llm=mock_llm)

    response = orchestrator.handle("Goodbye", [])
    assert "Goodbye" in response
    assert len(mock_llm.calls) == 0, "Farewell query must make 0 LLM calls"


def test_hello_and_all_okay_make_exactly_one_llm_call():
    """Verify greetings and acknowledgments make at most 1 LLM call with bounded max_tokens."""
    mock_llm = CountingMockLLM(response="Hello there! How can I help you today?")
    orchestrator = MAIOrchestrator(llm=mock_llm)

    # Turn A: Hello
    res_a = orchestrator.handle("Hello", [])
    assert "Hello" in res_a
    assert len(mock_llm.calls) == 1, "Greeting must make exactly 1 LLM call"
    assert mock_llm.calls[0]["stage"] == "conversation"
    assert mock_llm.calls[0]["max_tokens"] == 150, "Greeting must cap max_tokens to 150"

    # Turn B: All okay
    mock_llm.calls.clear()
    mock_llm.response = "Glad to hear that! Let me know if you need anything."
    res_b = orchestrator.handle("All okay", [])
    assert "Glad" in res_b
    assert len(mock_llm.calls) == 1, "Acknowledgment must make exactly 1 LLM call"
    assert mock_llm.calls[0]["stage"] == "conversation"
    assert mock_llm.calls[0]["max_tokens"] == 150, "Acknowledgment must cap max_tokens to 150"


def test_per_turn_accounting_in_resilient_provider():
    """Verify ResilientLLMProvider records call accounting, token usage, and stage tagging."""
    primary = CountingMockLLM(name="groq", response="response from groq")
    fallback = CountingMockLLM(name="ollama", response="response from ollama")

    resilient = ResilientLLMProvider(primary_provider=primary, fallback_provider=fallback)
    resilient.reset_turn_accounting()

    assert resilient.get_turn_accounting()["total_calls"] == 0

    res = resilient.generate([{"role": "user", "content": "hi"}], stage="conversation", max_tokens=150)
    assert res == "response from groq"

    accounting = resilient.get_turn_accounting()
    assert accounting["groq_call_count"] == 1
    assert accounting["total_calls"] == 1
    assert accounting["fallback_occurred"] is False
    assert accounting["total_prompt_tokens"] == 50
    assert accounting["total_completion_tokens"] == 20
    assert accounting["total_tokens"] == 70

    call_rec = accounting["calls"][0]
    assert call_rec["stage"] == "conversation"
    assert call_rec["provider"] == "groq"
    assert call_rec["http_status"] == "200_OK"


def test_429_rate_limit_fallback_and_accounting():
    """Verify ResilientLLMProvider handles HTTP 429 cleanly without looping and records telemetry."""
    primary = CountingFailLLM(retry_after=10.875)
    fallback = CountingMockLLM(name="ollama", response="local answer")

    resilient = ResilientLLMProvider(primary_provider=primary, fallback_provider=fallback)
    resilient.reset_turn_accounting()

    res = resilient.generate([{"role": "user", "content": "test"}], stage="conversation", max_tokens=150)
    assert res == "local answer"

    accounting = resilient.get_turn_accounting()
    assert accounting["total_calls"] == 2  # 1 attempted primary + 1 successful fallback
    assert accounting["fallback_occurred"] is True

    # Check failure record
    primary_call = accounting["calls"][0]
    assert primary_call["provider"] == "groq"
    assert "429" in primary_call["http_status"]
    assert "rate_limit_429" in primary_call["fallback_reason"]

    # Check fallback record
    fb_call = accounting["calls"][1]
    assert fb_call["provider"] == "ollama"
    assert fb_call["http_status"] == "200_OK"
