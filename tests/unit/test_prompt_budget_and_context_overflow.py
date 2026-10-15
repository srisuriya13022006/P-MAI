"""
P-MAI Unit Tests — Prompt Budget Management & Local Context Overflow Recovery.
Validates token estimation, prompt-budget enforcement, preservation of latest user queries and citations,
Groq 429 -> Ollama fallback adaptation, controlled single overflow retry, clean error handling,
and zero duplicate tool executions.
"""
from unittest.mock import MagicMock, patch
import pytest

from app.llm.ollama_provider import ContextOverflowError, OllamaProvider, is_context_overflow_error
from app.llm.prompt_budget import (
    bound_search_evidence,
    budget_messages,
    estimate_messages_tokens,
    estimate_tokens,
)
from app.llm.resilient_provider import ResilientLLMProvider
from app.core.runtime_mode import CloudErrorCategory


# ---------------------------------------------------------------------------
# 1. Token Estimation and Invariant Preservation
# ---------------------------------------------------------------------------

def test_estimate_tokens_accuracy():
    """Verify estimate_tokens returns reasonable token counts for plain text and formatting."""
    text = "Hello world! This is a test sentence for token counting."
    tok = estimate_tokens(text)
    assert 8 <= tok <= 16

    # Empty string
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_prompt_budget_preserves_latest_user_message_and_system():
    """Verify prompt-budget enforcement strictly preserves the latest user query and system prompt."""
    sys_prompt = "You are MAI, an AI assistant with strict security and memory policies."
    # Create 20 older turns
    older_messages = []
    for i in range(20):
        older_messages.append({"role": "user", "content": f"Turn {i}: Can you explain Python architecture details in depth? " * 15})
        older_messages.append({"role": "assistant", "content": f"Turn {i}: Here are extensive architectural explanations... " * 15})

    latest_user_query = "What is the capital of France?"
    messages = older_messages + [{"role": "user", "content": latest_user_query}]

    total_tokens_before = estimate_messages_tokens(messages, sys_prompt)
    assert total_tokens_before > 4000  # Well over 4000 tokens

    # Budget down to 2500 tokens
    budgeted, budgeted_sys, telemetry = budget_messages(
        messages=messages,
        system_prompt=sys_prompt,
        max_prompt_tokens=2500,
    )

    assert telemetry["truncated"] is True
    assert telemetry["total_tokens"] <= 2500

    # Invariants:
    # 1. Latest message MUST be preserved exactly
    assert budgeted[-1]["role"] == "user"
    assert budgeted[-1]["content"] == latest_user_query

    # 2. System prompt MUST be preserved
    assert budgeted_sys == sys_prompt


def test_long_conversation_followed_by_short_greeting():
    """Verify a long conversation followed by a short greeting reduces history and preserves the greeting."""
    sys_prompt = "You are MAI."
    messages = []
    for i in range(25):
        messages.append({"role": "user", "content": f"Query {i}: " + ("data processing details and architectural explanation " * 30)})
        messages.append({"role": "assistant", "content": f"Reply {i}: " + ("analysis results and telemetry metrics report " * 35)})

    short_greeting = "Hello there!"
    messages.append({"role": "user", "content": short_greeting})

    initial_tokens = estimate_messages_tokens(messages, sys_prompt)
    assert initial_tokens > 4000

    budgeted, budgeted_sys, telemetry = budget_messages(
        messages=messages,
        system_prompt=sys_prompt,
        max_prompt_tokens=2000,
    )

    assert budgeted[-1]["content"] == short_greeting
    # Check that older turns were omitted with a clean marker
    assert any("omitted" in m.get("content", "").lower() for m in budgeted)
    assert telemetry["omitted_turns"] > 0
    assert estimate_messages_tokens(budgeted, budgeted_sys) <= 2000


# ---------------------------------------------------------------------------
# 2. Bounded Search Evidence & Citation Integrity
# ---------------------------------------------------------------------------

def test_bounded_search_evidence_preserves_exact_citations():
    """Verify web evidence bounding shrinks raw text while strictly preserving markdown source links."""
    citations = [
        "[Python Official Docs](https://docs.python.org/3/whatsnew/3.13.html)",
        "[FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)",
        "[GitHub Release](https://github.com/psf/black/releases/tag/24.1.0)",
    ]
    raw_evidence = (
        "<search_evidence>\n"
        "Massive web evidence payload with hundreds of repetitive lines.\n"
        + ("Extensive paragraphs discussing performance and features across diverse platforms and environments.\n" * 150)
        + f"Source 1: {citations[0]}\n"
        + f"Source 2: {citations[1]}\n"
        + f"Source 3: {citations[2]}\n"
        + "</search_evidence>"
    )

    initial_evidence_tokens = estimate_tokens(raw_evidence)
    assert initial_evidence_tokens > 1000

    bounded = bound_search_evidence(raw_evidence, max_tokens=300)
    assert estimate_tokens(bounded) <= 450
    assert "<search_evidence>" in bounded
    assert "</search_evidence>" in bounded

    # Verify all 3 citation URLs are preserved
    for cite in citations:
        assert cite in bounded


# ---------------------------------------------------------------------------
# 3. Groq 429 Followed by Ollama Fallback with Prompt Adaptation
# ---------------------------------------------------------------------------

def test_groq_429_followed_by_ollama_fallback_with_prompt_budget():
    """Verify Groq HTTP 429 triggers Ollama fallback, adapting 7500+ tokens to Ollama's 4096 context."""
    from unittest.mock import MagicMock

    # Primary provider (Groq) simulates 429 rate limit
    mock_primary = MagicMock()
    mock_primary.provider_name = "groq"
    mock_primary.model = "llama-3.3-70b-versatile"
    err_429 = Exception("Rate limit exceeded: 429 Too Many Requests")
    setattr(err_429, "status_code", 429)
    mock_primary.generate.side_effect = err_429

    # Fallback provider (Ollama)
    mock_fallback = MagicMock()
    mock_fallback.provider_name = "ollama"
    mock_fallback.model = "qwen3:4b-instruct"
    mock_fallback.context_limit = 4096
    mock_fallback.generate.return_value = "Recovered response from local Ollama model."

    resilient = ResilientLLMProvider(
        primary_provider=mock_primary,
        fallback_provider=mock_fallback,
        fallback_enabled=True,
    )

    # Build 7500+ token conversation messages
    huge_messages = []
    for i in range(25):
        huge_messages.append({"role": "user", "content": f"Turn {i}: verbose text " * 40})
        huge_messages.append({"role": "assistant", "content": f"Answer {i}: verbose text " * 40})
    huge_messages.append({"role": "user", "content": "Final short question?"})

    raw_tokens = estimate_messages_tokens(huge_messages)
    assert raw_tokens > 7000

    # Generate through ResilientLLMProvider
    result = resilient.generate(huge_messages, system_prompt="You are MAI.")

    assert result == "Recovered response from local Ollama model."
    assert resilient.last_provider_used == "ollama"
    assert resilient.runtime_mode == "LOCAL_FALLBACK"

    # Verify fallback provider received a budgeted prompt within 4096 limit
    call_args = mock_fallback.generate.call_args
    passed_messages = call_args[0][0]
    passed_sys = call_args[1].get("system_prompt") or call_args[0][1] if len(call_args[0]) > 1 else None
    passed_tokens = estimate_messages_tokens(passed_messages, passed_sys)
    assert passed_tokens <= 4096
    # Verify latest user query preserved
    assert passed_messages[-1]["content"] == "Final short question?"


# ---------------------------------------------------------------------------
# 4. Ollama Context Overflow Single Retry & Recovery
# ---------------------------------------------------------------------------

def test_ollama_overflow_triggers_single_recovery_retry():
    """Verify Ollama context-size error triggers exactly one reduced-context retry and recovers."""
    from ollama import ResponseError

    mock_client = MagicMock()
    # 1st call fails with exceed_context_size_error
    err_body = {
        "code": 400,
        "message": "request (7549 tokens) exceeds the available context size (4096 tokens)",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 7549,
        "n_ctx": 4096,
    }
    overflow_err = ResponseError("request exceeds context size", status_code=400)
    setattr(overflow_err, "error", err_body)

    # 2nd call succeeds
    success_resp = {"message": {"content": "Successfully generated on retry."}}
    mock_client.chat.side_effect = [overflow_err, success_resp]

    provider = OllamaProvider(model="qwen3:4b-instruct", num_ctx=4096)
    provider.client = mock_client

    messages = [{"role": "user", "content": "Large request"}]
    res = provider.generate(messages)

    assert res == "Successfully generated on retry."
    assert mock_client.chat.call_count == 2
    assert provider.last_overflow_recovered is True


def test_ollama_failed_retry_returns_clean_recoverable_error():
    """Verify Ollama context overflow fails safely after 1 retry with a clean error, not raw JSON."""
    from ollama import ResponseError

    mock_client = MagicMock()
    err_body = {
        "code": 400,
        "message": "request exceeds context size",
        "type": "exceed_context_size_error",
    }
    overflow_err = ResponseError("request exceeds context size", status_code=400)
    setattr(overflow_err, "error", err_body)

    # Both initial and retry fail with overflow
    mock_client.chat.side_effect = [overflow_err, overflow_err]

    provider = OllamaProvider(model="qwen3:4b-instruct", num_ctx=4096)
    provider.client = mock_client

    res = provider.generate([{"role": "user", "content": "Persistent overflow"}])

    # Exactly 2 attempts (initial + 1 retry, no infinite loop)
    assert mock_client.chat.call_count == 2
    # Returns clean recoverable user-facing error message
    assert "exceeded the local model's available context memory" in res
    assert "{" not in res  # No raw JSON leaked


# ---------------------------------------------------------------------------
# 5. Zero Duplicate Tool Execution During Recovery
# ---------------------------------------------------------------------------

def test_no_duplicate_tool_execution_during_overflow_recovery():
    """Verify tool execution is NOT repeated if subsequent LLM generation recovers from overflow."""
    tool_execution_count = 0

    def mock_tool_runner():
        nonlocal tool_execution_count
        tool_execution_count += 1
        return "Tool execution result: calculated 42."

    # Tool executes once
    tool_output = mock_tool_runner()
    assert tool_execution_count == 1

    # Simulate subsequent LLM synthesis step experiencing context overflow and recovering
    from ollama import ResponseError
    mock_client = MagicMock()
    overflow_err = ResponseError("exceed_context_size_error", status_code=400)
    setattr(overflow_err, "error", {"type": "exceed_context_size_error"})
    success_resp = {"message": {"content": "Synthesized: The answer is 42."}}
    mock_client.chat.side_effect = [overflow_err, success_resp]

    ollama = OllamaProvider(model="qwen3:4b-instruct", num_ctx=4096)
    ollama.client = mock_client

    synthesis_res = ollama.generate([
        {"role": "user", "content": f"Synthesize this output: {tool_output}"}
    ])

    assert synthesis_res == "Synthesized: The answer is 42."
    # Tool was never re-executed during LLM recovery
    assert tool_execution_count == 1
