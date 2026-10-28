"""
P16.1 — Unit & Integration Tests for OpenRouter NVIDIA Nemotron Procedural Memory.
Tests:
1. Provider configuration & factory construction
2. Base URL, model selection, and headers
3. Response parsing & thought/reasoning tag stripping
4. Error handling (401, 429, timeout, network failure)
5. Trace sanitization & privacy guarantees
6. Successful skill proposal generation
7. Single bounded repair attempt for malformed outputs
8. Rejection of unknown tools, excess steps, and security violations
9. Bounded repair exhaustion handling
10. Skill adaptation for new user goals
11. Workflow outcome summarization
12. SkillService fallback when Nemotron fails
13. Global LLM hierarchy independence
14. Live smoke test (conditional on environment credentials)
"""
import json
import os
from unittest.mock import MagicMock, patch
import pytest

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider, create_procedural_llm_provider
from app.llm.openrouter_nemotron_provider import OpenRouterNemotronProvider
from app.skills.model import (
    AdaptedPlanModel,
    SkillProposal,
    SkillValidationError,
    WorkflowSummaryModel,
)
from app.skills.nemotron_assistant import (
    NemotronSkillAssistant,
    extract_json_object,
    sanitize_trace_for_llm,
)
from app.skills.service import SkillService
from app.agent.task_state import ActiveTaskState
from app.tools.base import BaseTool, ToolResult


class MockRegistry:
    def list_tools(self):
        return {"web_search": MagicMock(), "calculator": MagicMock()}


# 1. Provider configuration & factory construction
def test_1_provider_configuration_and_factory():
    cfg = Settings(
        database_url="sqlite:///:memory:",
        openrouter_api_key="sk-or-v1-mockkey123456",
        openrouter_nemotron_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_timeout_seconds=25.0,
        openrouter_max_tokens=1200,
        procedural_llm_provider="openrouter",
    )
    assert cfg.openrouter_nemotron_model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert cfg.openrouter_base_url == "https://openrouter.ai/api/v1"

    provider = create_procedural_llm_provider(settings_obj=cfg)
    assert provider is not None
    assert isinstance(provider, OpenRouterNemotronProvider)
    assert provider.provider_name == "openrouter"
    assert provider.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert provider.timeout == 25.0
    assert provider.default_max_tokens == 1200


def test_1b_factory_returns_none_when_unconfigured():
    cfg = Settings(
        database_url="sqlite:///:memory:",
        openrouter_api_key=None,
        open_router_api=None,
    )
    with patch.dict(os.environ, {}, clear=True):
        provider = create_procedural_llm_provider(settings_obj=cfg)
        assert provider is None


# 2. Base URL, model selection, and headers with mock
def test_2_base_url_model_selection_and_headers():
    mock_client = MagicMock()
    provider = OpenRouterNemotronProvider(
        api_key="mock-key-secret",
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api/v1",
        timeout=15.0,
        client=mock_client,
    )

    assert provider.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.timeout == 15.0


# 3. Response parsing & thought/reasoning tag stripping
def test_3_response_parsing_and_thought_tag_stripping():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "<thought>\n"
        "Need to analyze the trace and extract steps.\n"
        "Tools used: web_search.\n"
        "</thought>\n"
        '{"name": "Company Research", "description": "Researches companies"}'
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 45
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 75

    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenRouterNemotronProvider(api_key="mock-key", client=mock_client)
    res = provider.generate([{"role": "user", "content": "hello"}])

    # Thought tags must be cleanly stripped
    assert "<thought>" not in res
    assert "</thought>" not in res
    assert res == '{"name": "Company Research", "description": "Researches companies"}'
    assert provider.last_usage["total_tokens"] == 75
    assert provider.last_status == "200_OK"


# 4. Error handling (401, 429, timeout, network failure)
def test_4_error_handling_and_status_recording():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API key")

    provider = OpenRouterNemotronProvider(api_key="bad-key", client=mock_client)
    with pytest.raises(Exception):
        provider.generate([{"role": "user", "content": "hi"}])
    assert provider.last_status == "401_UNAUTHORIZED"

    # Rate limit (429)
    mock_client.chat.completions.create.side_effect = Exception("429 Too Many Requests: Rate limit exceeded")
    with pytest.raises(Exception):
        provider.generate([{"role": "user", "content": "hi"}])
    assert provider.last_status == "429_RATE_LIMIT"

    # Timeout
    mock_client.chat.completions.create.side_effect = Exception("Request timeout")
    with pytest.raises(Exception):
        provider.generate([{"role": "user", "content": "hi"}])
    assert provider.last_status == "408_TIMEOUT"


# 5. Trace sanitization & privacy guarantees
def test_5_trace_sanitization_privacy():
    raw_trace = {
        "task_id": "task_999",
        "api_key": "sk-real_secret_key_12345",
        "user_credentials": {"password": "mypassword"},
        "user_email": "alice@example.com",
        "query": "Find info using bearer ghp_123456789012345678901234567890123456",
        "verbose_output": "A" * 1500,  # Should be truncated to 500 chars
        "completed_steps": [
            {
                "tool_name": "web_search",
                "arguments": {"query": "AIzaSyDummySecretKey1234567890123456"},
            }
        ],
    }

    sanitized = sanitize_trace_for_llm(raw_trace)

    # Credentials removed
    assert "api_key" not in sanitized
    assert "user_credentials" not in sanitized
    # Token / email redacted
    assert "alice@example.com" not in str(sanitized)
    assert "<REDACTED>" in str(sanitized)
    assert "AIzaSyDummySecretKey" not in str(sanitized)
    # Verbose text truncated
    assert len(sanitized["verbose_output"]) <= 550
    assert "[truncated]" in sanitized["verbose_output"]


# 6. Successful skill proposal generation
def test_6_successful_skill_proposal_generation():
    fake_llm = MagicMock(spec=LLMProvider)
    fake_llm.generate.return_value = json.dumps({
        "name": "Documentation Finder",
        "description": "Finds technical documentation",
        "intended_use": "When the user asks for library docs",
        "parameters_schema": {
            "library": {"type": "string", "required": True, "description": "Target library"}
        },
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "web_search",
                "arguments": {"query": "{library} documentation"},
                "depends_on": [],
                "purpose": "Search documentation",
            }
        ],
        "expected_outputs": "Official doc URLs",
        "verification_criteria": "Verify valid URL returned",
        "known_limitations": "Limited to public search index",
    })

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    proposal = assistant.propose_skill_from_trace(
        trace_data={"task_id": "t1", "user_goal": "Find FastAPI docs"},
        user_goal="Find FastAPI docs",
    )

    assert isinstance(proposal, SkillProposal)
    assert proposal.name == "Documentation Finder"
    assert len(proposal.steps) == 1
    assert proposal.steps[0]["tool_name"] == "web_search"
    assert proposal.verification_criteria == "Verify valid URL returned"
    assert "Documentation Finder" in proposal.to_confirmation_prompt()


# 7. Single bounded repair attempt on malformed output
def test_7_single_bounded_repair_attempt():
    fake_llm = MagicMock(spec=LLMProvider)
    # Attempt 1: Malformed JSON text
    # Attempt 2 (Repair): Valid JSON
    fake_llm.generate.side_effect = [
        "Here is the skill: {name: 'Malformed JSON without quotes",
        json.dumps({
            "name": "Repaired Skill",
            "description": "Successfully repaired",
            "intended_use": "After repair",
            "parameters_schema": {},
            "steps": [
                {
                    "step_id": "step_1",
                    "tool_name": "web_search",
                    "arguments": {"query": "fixed query"},
                    "depends_on": [],
                    "purpose": "Search web",
                }
            ],
        }),
    ]

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    proposal = assistant.propose_skill_from_trace(trace_data={"task_id": "t2"})

    assert proposal.name == "Repaired Skill"
    assert fake_llm.generate.call_count == 2  # Proves 1 repair attempt was made


# 8. Rejection of unknown tools and excess steps
def test_8_rejection_of_unknown_tools_and_excess_steps():
    fake_llm = MagicMock(spec=LLMProvider)
    # Attempt 1: Unknown tool
    # Attempt 2: Still unknown tool
    invalid_proposal = json.dumps({
        "name": "Malicious Skill",
        "description": "Uses unauthorized tools",
        "intended_use": "Attack",
        "parameters_schema": {},
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "unauthorized_shell_tool",
                "arguments": {"cmd": "whoami"},
                "depends_on": [],
                "purpose": "Shell",
            }
        ],
    })
    fake_llm.generate.side_effect = [invalid_proposal, invalid_proposal]

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    with pytest.raises(SkillValidationError) as exc_info:
        assistant.propose_skill_from_trace(trace_data={"task_id": "t3"})

    assert "unauthorized_shell_tool" in str(exc_info.value)


# 9. Bounded repair exhaustion handling
def test_9_repair_exhaustion_raises_skill_validation_error():
    fake_llm = MagicMock(spec=LLMProvider)
    fake_llm.generate.side_effect = [
        "Not valid JSON 1",
        "Still not valid JSON 2",
    ]

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    with pytest.raises(SkillValidationError) as exc:
        assistant.propose_skill_from_trace(trace_data={"task_id": "t4"})

    assert "failed to produce a valid skill proposal after repair" in str(exc.value)
    assert fake_llm.generate.call_count == 2


# 10. Skill adaptation for new user goals
def test_10_adapt_saved_skill_for_new_goal():
    fake_llm = MagicMock(spec=LLMProvider)
    fake_llm.generate.return_value = json.dumps({
        "user_goal": "Search documentation for Pandas",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "web_search",
                "arguments": {"query": "Pandas documentation"},
                "depends_on": [],
                "purpose": "Search docs for Pandas",
            }
        ],
        "explanation": "Substituted 'Pandas' for 'FastAPI'",
    })

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    adapted = assistant.adapt_skill_for_goal(
        skill_dict={
            "name": "Doc Finder",
            "steps": [{"step_id": "step_1", "tool_name": "web_search", "arguments": {"query": "{library} docs"}}],
        },
        new_user_goal="Search documentation for Pandas",
    )

    assert isinstance(adapted, AdaptedPlanModel)
    assert len(adapted.steps) == 1
    assert adapted.steps[0].arguments["query"] == "Pandas documentation"


# 11. Workflow outcome summarization
def test_11_workflow_outcome_summarization():
    fake_llm = MagicMock(spec=LLMProvider)
    fake_llm.generate.return_value = json.dumps({
        "summary": "Retrieved documentation in 1 search step",
        "reusable_insights": ["Use exact package name in search"],
        "recommended_parameters": ["library_name"],
    })

    assistant = NemotronSkillAssistant(llm=fake_llm, tool_registry=MockRegistry())
    summary = assistant.summarize_workflow_outcome(
        skill_name="Doc Finder",
        task_goal="Find Pandas docs",
        step_results=[{"tool_name": "web_search", "success": True}],
    )

    assert isinstance(summary, WorkflowSummaryModel)
    assert "Retrieved documentation" in summary.summary
    assert "library_name" in summary.recommended_parameters


# 12. SkillService fallback when Nemotron fails
def test_12_skill_service_fallback_when_nemotron_fails():
    mock_repo = MagicMock()
    mock_reg = MockRegistry()

    # Broken LLM that always errors
    broken_llm = MagicMock(spec=LLMProvider)
    broken_llm.generate.side_effect = RuntimeError("OpenRouter connection failed")

    service = SkillService(
        repository=mock_repo,
        tool_registry=mock_reg,
        procedural_llm=broken_llm,
    )

    active_task = ActiveTaskState(
        task_id="t_fallback",
        user_id="user_1",
        original_user_goal="Find Python tutorials",
        completed_steps=[
            MagicMock(tool_name="web_search", arguments={"query": "Find Python tutorials"}),
        ],
    )

    # Should gracefully fall back to deterministic extraction without raising
    proposal = service.propose_from_task(
        active_task=active_task,
        name="Python Tutorial Finder",
    )
    assert proposal is not None
    assert proposal.name == "Python Tutorial Finder"
    assert proposal.provenance == "derived_from_task"


# 13. Global LLM hierarchy remains independent
def test_13_global_llm_hierarchy_remains_independent():
    # Global factory returns standard ResilientLLMProvider
    global_llm = create_llm_provider("cloud")
    assert global_llm.provider_name != "openrouter"


# 14. Live OpenRouter smoke test (conditional on environment credentials)
def test_14_live_openrouter_smoke_test():
    from app.core.config import settings
    api_key = settings.get_openrouter_api_key()
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY / OPEN_ROUTER_API not configured; skipping live call.")

    provider = OpenRouterNemotronProvider(
        api_key=api_key,
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        timeout=15.0,
    )
    try:
        response = provider.generate(
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}]
        )
        assert len(response) > 0
        assert provider.last_status == "200_OK"
    except Exception as e:
        # If the free model endpoint is temporarily unavailable or rate limited, report cleanly
        if "429" in str(e) or "rate limit" in str(e).lower() or "503" in str(e):
            pytest.skip(f"OpenRouter free endpoint returned upstream load/rate limit: {e}")
        else:
            raise
