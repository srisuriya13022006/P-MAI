"""
P15.9 Unit Tests: Conversational Intelligence, Context Awareness & Personality Refinement.
Validates:
1. Conversational follow-up resolution:
   - Resolves "Hey, answer the question!" and "Answer directly" to preceding question
   - Follow-up varieties: "What do you mean?", "Explain that", "What about you?", "Tell me more", "Compare them directly", "Read it again"
2. Direct answer first policy:
   - Direct conclusion stated upfront for comparisons, yes/no, and capability questions
3. No-Sir / Anti-servility policy:
   - Never addresses the user as "Sir", "Boss", or "Master"
4. Implemented capabilities and tools explanation:
   - Explains real tools (web search, web fetch, calculator, datetime, memory, file search, voice)
5. Creative generation & song token headroom:
   - Allocates 1000 tokens and cloud model eligibility for songs, lyrics, ballads, poems, stories
6. Ambiguous follow-up handling & no-prior-context guards:
   - Asks concise clarification when prior context is missing or genuinely ambiguous
   - Never invents earlier conversation content
   - Preserves explicit topic changes
"""
from unittest.mock import MagicMock
import pytest

from app.agent.orchestrator import MAIOrchestrator
from app.agent.reference_resolver import (
    extract_recent_conversation_context,
    resolve_task_reference,
)
from app.prompts.conversation import MAI_SYSTEM_PROMPT
from app.schemas.agent import AgentDecision


# ---------------------------------------------------------------------------
# 1. Follow-up Context Extraction and Reference Resolution Tests
# ---------------------------------------------------------------------------

def test_extract_recent_conversation_context():
    history = [
        {"role": "user", "content": "Are you equal to Jarvis?"},
        {"role": "assistant", "content": "Jarvis is a fictional AI from Marvel comics with vast cinematic capabilities."},
        {"role": "user", "content": "Hey, answer the question!"},
    ]

    last_user, last_assistant = extract_recent_conversation_context("Hey, answer the question!", history)
    assert last_user == "Are you equal to Jarvis?"
    assert "Jarvis is a fictional AI" in last_assistant


def test_resolve_direct_answer_followup_with_history():
    history = [
        {"role": "user", "content": "Are you equal to Jarvis?"},
        {"role": "assistant", "content": "Jarvis is portrayed with fictional abilities in the movies."},
        {"role": "user", "content": "Hey, answer the question!"},
    ]

    res = resolve_task_reference("Hey, answer the question!", None, conversation_messages=history)
    assert res.is_reference_resolved is True
    assert res.contextual_followup_type == "direct_answer"
    assert res.original_question == "Are you equal to Jarvis?"
    assert "Answer directly: Are you equal to Jarvis?" in res.resolved_message


def test_resolve_direct_answer_followup_varieties():
    history = [
        {"role": "user", "content": "Can you hack into a satellite?"},
        {"role": "assistant", "content": "Satellite communication involves complex protocols."},
    ]

    for phrase in ("Answer directly.", "You didn't answer my question", "Answer the question", "Just answer"):
        res = resolve_task_reference(phrase, None, conversation_messages=history)
        assert res.is_reference_resolved is True
        assert res.contextual_followup_type == "direct_answer"
        assert res.original_question == "Can you hack into a satellite?"


def test_resolve_explain_followup():
    history = [
        {"role": "user", "content": "Is Python better than C++?"},
        {"role": "assistant", "content": "Python trades execution speed for developer ergonomics."},
        {"role": "user", "content": "What do you mean?"},
    ]

    res = resolve_task_reference("What do you mean?", None, conversation_messages=history)
    assert res.is_reference_resolved is True
    assert res.contextual_followup_type == "explain"
    assert "developer ergonomics" in res.resolved_message or "Python trades" in res.resolved_message


def test_resolve_pivot_followup():
    history = [
        {"role": "user", "content": "Jarvis can fly the Iron Man suit."},
        {"role": "assistant", "content": "Jarvis controls physical armor in Marvel films."},
        {"role": "user", "content": "What about you?"},
    ]

    res = resolve_task_reference("What about you?", None, conversation_messages=history)
    assert res.is_reference_resolved is True
    assert res.contextual_followup_type == "pivot"
    assert "MAI" in res.resolved_message


def test_resolve_compare_directly_followup():
    history = [
        {"role": "user", "content": "Who is more capable, Jarvis or you?"},
        {"role": "assistant", "content": "There are key differences between fictional movie AI and real assistants."},
        {"role": "user", "content": "Compare them directly."},
    ]

    res = resolve_task_reference("Compare them directly.", None, conversation_messages=history)
    assert res.is_reference_resolved is True
    assert res.contextual_followup_type == "compare"


def test_resolve_repeat_followup():
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Read it again."},
    ]

    res = resolve_task_reference("Read it again.", None, conversation_messages=history)
    assert res.is_reference_resolved is True
    assert res.target_tool == "repeat"
    assert res.resolved_message == "The capital of France is Paris."


# ---------------------------------------------------------------------------
# 2. Ambiguity & No-Prior-Context Guards
# ---------------------------------------------------------------------------

def test_followup_without_prior_context_asks_clarification():
    # If the user says "Answer the question!" with NO conversation history, must not crash or invent facts
    res = resolve_task_reference("Hey, answer the question!", None, conversation_messages=[])
    assert res.is_reference_resolved is False
    assert res.needs_clarification is True
    assert "Which question would you like me to answer?" in res.clarification_prompt


def test_explain_without_prior_context_asks_clarification():
    res = resolve_task_reference("What do you mean?", None, conversation_messages=[])
    assert res.needs_clarification is True
    assert "What would you like me to explain?" in res.clarification_prompt


def test_explicit_topic_change_is_not_overridden():
    history = [
        {"role": "user", "content": "Are you equal to Jarvis?"},
        {"role": "assistant", "content": "Jarvis has cinematic abilities."},
        {"role": "user", "content": "What is 15 * 4?"},
    ]

    # Explicit calculation is a new topic, NOT a follow-up
    res = resolve_task_reference("What is 15 * 4?", None, conversation_messages=history)
    assert res.is_reference_resolved is False
    assert res.needs_clarification is False
    assert res.resolved_message == "What is 15 * 4?"


# ---------------------------------------------------------------------------
# 3. Direct-Answer-First and Orchestrator Contextual Steering Tests
# ---------------------------------------------------------------------------

def test_orchestrator_handles_followup_to_preceding_question():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "No, not yet. Jarvis, as portrayed in the movies, has fictional capabilities far beyond mine."

    orchestrator = MAIOrchestrator(llm=mock_llm)

    history = [
        {"role": "user", "content": "Are you equal to Jarvis?"},
        {"role": "assistant", "content": "Jarvis is an advanced fictional AI from Iron Man with capabilities in holographic interfaces and autonomous robotics."},
        {"role": "user", "content": "Hey, answer the question!"},
    ]

    response = orchestrator.handle(
        user_message="Hey, answer the question!",
        conversation_messages=history,
    )

    assert "No, not yet" in response
    assert mock_llm.generate.call_count >= 1
    # Verify that the conversation LLM call received contextual instructions demanding a direct answer to the preceding question
    called_messages = mock_llm.generate.call_args_list[-1][0][0]
    last_prompt_msg = called_messages[-1]["content"]
    assert "Are you equal to Jarvis?" in last_prompt_msg
    assert "direct" in last_prompt_msg.lower()


def test_orchestrator_handles_repeat_without_llm():
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(llm=mock_llm)

    history = [
        {"role": "user", "content": "Where is the Eiffel Tower?"},
        {"role": "assistant", "content": "The Eiffel Tower is located in Paris, France on the Champ de Mars."},
        {"role": "user", "content": "Read it again."},
    ]

    response = orchestrator.handle(
        user_message="Read it again.",
        conversation_messages=history,
    )

    assert response == "The Eiffel Tower is located in Paris, France on the Champ de Mars."
    mock_llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Anti-Servility (No "Sir") Policy Tests
# ---------------------------------------------------------------------------

def test_system_prompt_forbids_sir_and_servile_honorifics():
    assert "NEVER address the user as \"Sir\"" in MAI_SYSTEM_PROMPT
    assert "peer-to-peer" in MAI_SYSTEM_PROMPT.lower()
    assert "butler" in MAI_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# 5. Explaining Implemented Tools & Capabilities
# ---------------------------------------------------------------------------

def test_orchestrator_explains_all_implemented_tools():
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(llm=mock_llm)

    for query in (
        "What tools do you have?",
        "What are your tools?",
        "What tools are implemented?",
        "What can you do?",
        "Explain your capabilities.",
    ):
        res = orchestrator.handle(user_message=query, conversation_messages=[])
        assert "I'm MAI" in res
        assert "web search" in res.lower()
        assert "web fetch" in res.lower()
        assert "calculator" in res.lower()
        assert "date" in res.lower() or "time" in res.lower()
        assert "memory" in res.lower()
        assert "voice" in res.lower()
        mock_llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Song & Creative Headroom Verification
# ---------------------------------------------------------------------------

def test_creative_song_and_poem_token_headroom():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[Verse 1] Cold winds blow on the rocky shore..."
    orchestrator = MAIOrchestrator(llm=mock_llm)

    # Short query for a song (< 18 words)
    orchestrator.handle(
        user_message="Write a song about the sea",
        conversation_messages=[],
    )

    assert mock_llm.generate.call_count >= 1
    kwargs = mock_llm.generate.call_args_list[-1][1]
    # Must allocate 1000 tokens, not the short-message cap of 300 tokens
    assert kwargs.get("max_tokens") == 1000


def test_creative_should_use_cloud_model():
    mock_llm = MagicMock()
    orchestrator = MAIOrchestrator(llm=mock_llm)
    orchestrator.cloud_llm = MagicMock()

    decision = AgentDecision(
        intent="creative",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="Creative generation query",
    )

    for creative_prompt in (
        "Write a song about Arthur's lighthouse",
        "Compose lyrics for an anthem",
        "Write a ballad about the storm",
        "Write a poem about winter",
    ):
        assert orchestrator._should_use_cloud_model(creative_prompt, decision) is True
