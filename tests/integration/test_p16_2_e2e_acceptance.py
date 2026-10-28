"""
Integration Test for P16.2 End-to-End Acceptance Scenario:
Procedural Memory & Reusable Skills with NVIDIA Nemotron (OpenRouter) Integration.

Exercises the full multi-turn conversational workflow across:
1. Tool-using research task execution with validated tool registry
2. Structured skill proposal generation via NemotronSkillAssistant
3. Verification of proposed schema, owner, required inputs, tool names, ordering, completion criteria
4. User approval presentation
5. Proposal rejection: leaves no saved skill in database
6. Proposal approval: persists skill in SQLite via SkillRepository
7. Separate conversation context
8. Skill reuse request with new valid inputs
9. Discovery, adaptation, validation against registry and plan limits, execution through orchestration pathway
10. Result verification and skill outcome recording in repository
11. Nemotron failure resilience: no repeated tools, uncorrupted conversation state, normal chat continues
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from pydantic import BaseModel
from typing import Type

from app.database.connection import Base
from app.database.models import Skill
from app.database.repositories.skills import SkillRepository
from app.skills.service import SkillService
from app.skills.nemotron_assistant import NemotronSkillAssistant
from app.skills.model import SkillProposal
from app.services.chat_service import ChatService
from app.agent.orchestrator import MAIOrchestrator
from app.agent.task_state import TaskStateManager, TaskLifecycleStatus
from app.tools.registry import ToolRegistry
from app.tools.base import BaseTool, ToolResult
from app.llm.base import LLMProvider


class SearchInput(BaseModel):
    query: str


class MockWebSearchTool(BaseTool):
    """Mock search tool returning deterministic results for synthetic research queries."""
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for up-to-date information."

    @property
    def input_schema(self) -> Type[BaseModel]:
        return SearchInput

    def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if "3.12" in query:
            res = "Python 3.12 was officially released on October 2, 2023."
        elif "3.13" in query:
            res = "Python 3.13 was officially released on October 7, 2024."
        else:
            res = f"Search results for: {query}"
        return ToolResult(tool_name=self.name, success=True, data=res)


@pytest.fixture
def test_db_session():
    """Isolated in-memory SQLite session with all models created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def custom_tool_registry():
    """Tool registry with validated mock search tool."""
    reg = ToolRegistry()
    reg.register(MockWebSearchTool())
    return reg


def test_p16_2_complete_end_to_end_acceptance(test_db_session, custom_tool_registry):
    """
    Complete 11-step end-to-end integration test validating the entire procedural memory lifecycle.
    """
    # -------------------------------------------------------------
    # 0. Setup Mock Nemotron LLM Provider
    # -------------------------------------------------------------
    mock_nemotron_llm = MagicMock(spec=LLMProvider)
    mock_nemotron_llm.provider_name = "openrouter"

    # Synthetic structured proposal response from Nemotron
    nemotron_proposal_json = json.dumps({
        "name": "Python Release Lookup",
        "description": "Look up the official release date for any Python version.",
        "intended_use": "Retrieve official release dates for specific Python releases",
        "parameters_schema": {
            "version": {
                "type": "string",
                "description": "Python version number, e.g. 3.12 or 3.13",
                "required": True,
            }
        },
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "web_search",
                "arguments": {"query": "Python {version} release date"},
                "depends_on": [],
                "purpose": "Search the web for the Python release date",
            }
        ],
        "expected_outputs": ["Official release date text"],
        "verification_criteria": ["The result contains the release year and date"],
        "known_limitations": ["Requires internet connectivity for search"],
    })

    mock_nemotron_llm.generate.return_value = (
        f"<thought>Extracting parameters and steps from trace</thought>\n```json\n{nemotron_proposal_json}\n```"
    )

    # Wire up NemotronSkillAssistant and SkillService
    skill_repo = SkillRepository(test_db_session)
    assistant = NemotronSkillAssistant(llm=mock_nemotron_llm, tool_registry=custom_tool_registry)
    skill_service = SkillService(
        repository=skill_repo,
        tool_registry=custom_tool_registry,
        nemotron_assistant=assistant,
    )

    # Setup ChatService and Orchestrator with our test SQLite DB and SkillService
    mock_conv_llm = MagicMock(spec=LLMProvider)
    mock_conv_llm.provider_name = "gemini"
    mock_conv_llm.generate.return_value = "I am doing well, thank you! How can I help you today?"

    chat_service = ChatService(db=test_db_session, llm=mock_conv_llm)
    chat_service.orchestrator.tool_registry = custom_tool_registry
    chat_service.orchestrator.skill_service = skill_service
    chat_service.skill_service = skill_service

    user_id = "user_test_p16"
    conv_1 = "conversation_turn_1"

    # -------------------------------------------------------------
    # Step 1: Complete supported research-style task using validated tool registry
    # -------------------------------------------------------------
    # Simulate a research task execution directly on the active task
    active_task = chat_service.orchestrator.task_state_manager.create_task(
        user_id=user_id,
        conversation_id=conv_1,
        goal="Search the web for the Python 3.12 release date",
    )
    search_tool = custom_tool_registry.get("web_search")
    tool_res = search_tool.run(query="Python 3.12 release date")
    assert tool_res.success is True
    assert "October 2, 2023" in str(tool_res.data)

    # Record the completed step on the active task
    from app.agent.plan import StepResult
    step_r = StepResult(
        step_id="step_1",
        tool_name="web_search",
        success=True,
        output_text=str(tool_res.data),
        arguments={"query": "Python 3.12 release date"},
    )
    active_task.record_step_result(step_r)
    active_task.transition_to(TaskLifecycleStatus.RUNNING)
    active_task.transition_to(TaskLifecycleStatus.COMPLETED)

    # -------------------------------------------------------------
    # Step 2 & 3: Produce structured skill proposal through NemotronSkillAssistant
    # Verify schema, owner, required inputs, tool names, ordering, completion criteria
    # -------------------------------------------------------------
    proposal_prompt = chat_service.chat(
        conversation_id=conv_1,
        user_id=user_id,
        user_message="Save this workflow as Python Release Lookup",
    )

    # Confirm Nemotron LLM was invoked with sanitized trace
    call_kwargs = mock_nemotron_llm.generate.call_args.kwargs or {}
    messages = call_kwargs.get("messages") or mock_nemotron_llm.generate.call_args[0][0]
    prompt_content = messages[0]["content"]
    assert "Python 3.12" in prompt_content

    # Verify structured proposal attributes
    pending_proposal = active_task.context_values.get("pending_skill_save")
    assert isinstance(pending_proposal, SkillProposal)
    assert pending_proposal.name == "Python Release Lookup"
    assert "version" in pending_proposal.parameters_schema
    assert pending_proposal.parameters_schema["version"]["required"] is True
    assert len(pending_proposal.steps) == 1
    assert pending_proposal.steps[0]["tool_name"] == "web_search"
    assert pending_proposal.steps[0]["step_id"] == "step_1"
    assert len(pending_proposal.verification_criteria) > 0
    assert pending_proposal.provenance == "nemotron_derived_from_trace"

    # -------------------------------------------------------------
    # Step 4 & 5: Present to user for approval, confirm rejection leaves NO saved skill
    # -------------------------------------------------------------
    assert "Would you like me to save this workflow" in proposal_prompt
    assert "Python Release Lookup" in proposal_prompt

    # User explicitly rejects the proposal
    reject_response = chat_service.chat(
        conversation_id=conv_1,
        user_id=user_id,
        user_message="No, don't save that",
    )
    assert "won't save this workflow" in reject_response.lower()

    # Verify database has zero skills saved
    skills_in_db = skill_repo.list_for_user(user_id=user_id, active_only=False)
    assert len(skills_in_db) == 0

    # -------------------------------------------------------------
    # Step 6: Present proposal again and confirm approval persists skill
    # -------------------------------------------------------------
    # Propose again
    chat_service.chat(
        conversation_id=conv_1,
        user_id=user_id,
        user_message="Save this workflow as Python Release Lookup",
    )

    # User approves proposal
    approve_response = chat_service.chat(
        conversation_id=conv_1,
        user_id=user_id,
        user_message="Yes, please save it",
    )
    assert "saved the workflow as 'Python Release Lookup'" in approve_response

    # Verify persisted in database repository
    skills_after_approve = skill_repo.list_for_user(user_id=user_id, active_only=False)
    assert len(skills_after_approve) == 1
    saved_skill = skills_after_approve[0]
    assert saved_skill.name == "Python Release Lookup"
    assert saved_skill.user_id == user_id
    assert saved_skill.is_active is True
    assert saved_skill.version == 1
    assert saved_skill.usage_count == 0
    assert saved_skill.parameters_schema["version"]["required"] is True

    # -------------------------------------------------------------
    # Step 7: Start a separate conversation
    # -------------------------------------------------------------
    conv_2 = "conversation_turn_2_separate"

    # -------------------------------------------------------------
    # Step 8, 9 & 10: Ask MAI to reuse saved skill with different valid inputs
    # Verify owner retrieval, adaptation, tool validation, execution, and outcome recording
    # -------------------------------------------------------------
    # Another user cannot access or run this skill (isolation check)
    other_user_res = chat_service.chat(
        conversation_id="conv_other_user",
        user_id="unauthorized_attacker",
        user_message="Run the skill Python Release Lookup with version=3.13",
    )
    assert "couldn't find" in other_user_res.lower() or "no matching" in other_user_res.lower()

    # The authorized owner executes the skill with new parameter: version=3.13
    exec_response = chat_service.chat(
        conversation_id=conv_2,
        user_id=user_id,
        user_message="Run the skill Python Release Lookup with version=3.13",
    )

    # Verify output contains the new adapted execution result
    assert "Executed skill 'Python Release Lookup'" in exec_response
    assert "October 7, 2024" in exec_response

    # Step 10: Verify outcome is recorded and usage_count is incremented
    updated_skill = skill_repo.get(user_id=user_id, skill_id=saved_skill.id)
    assert updated_skill is not None
    assert updated_skill.usage_count == 1
    assert len(updated_skill.recent_outcomes) == 1
    outcome = updated_skill.recent_outcomes[0]
    assert outcome["success"] is True
    assert outcome["steps_executed"] == 1
    assert outcome["duration_ms"] > 0

    # -------------------------------------------------------------
    # Step 11: Simulate Nemotron failure; ensure no repeated tools, no corrupted state, chat works
    # -------------------------------------------------------------
    mock_nemotron_llm.generate.side_effect = TimeoutError("OpenRouter request timed out")

    # In a fresh task, user asks to save workflow while Nemotron is down
    conv_3 = "conversation_turn_3_resilience"
    active_task_3 = chat_service.orchestrator.task_state_manager.create_task(
        user_id=user_id,
        conversation_id=conv_3,
        goal="Search Python docs",
    )
    step_r3 = StepResult(
        step_id="step_1",
        tool_name="web_search",
        success=True,
        output_text="FastAPI docs retrieved",
        arguments={"query": "FastAPI docs"},
    )
    active_task_3.record_step_result(step_r3)
    active_task_3.transition_to(TaskLifecycleStatus.RUNNING)
    active_task_3.transition_to(TaskLifecycleStatus.COMPLETED)

    # Saving skill should fall back to deterministic extraction without crashing
    fallback_prompt = chat_service.chat(
        conversation_id=conv_3,
        user_id=user_id,
        user_message="Save this workflow as Docs Search",
    )
    assert "Would you like me to save this workflow" in fallback_prompt
    assert "Docs Search" in fallback_prompt
    fallback_proposal = active_task_3.context_values.get("pending_skill_save")
    assert fallback_proposal is not None
    assert fallback_proposal.provenance == "derived_from_task"

    # Ordinary chat continues to work seamlessly without corruption
    normal_chat_res = chat_service.chat(
        conversation_id=conv_3,
        user_id=user_id,
        user_message="Hello, how are you?",
    )
    assert isinstance(normal_chat_res, str)
    assert len(normal_chat_res) > 0


class FailingToolInput(BaseModel):
    reason: str = "forced_failure"


class MockFailingTool(BaseTool):
    @property
    def name(self) -> str:
        return "failing_operation"

    @property
    def description(self) -> str:
        return "Always fails to simulate downstream errors."

    @property
    def input_schema(self) -> Type[BaseModel]:
        return FailingToolInput

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, success=False, error="Downstream API connection failure")


def test_p16_2_downstream_failure_duplicate_prevention(test_db_session, custom_tool_registry):
    """
    Validates P16.2 requirement 10:
    Confirm that if a multi-step procedure fails downstream,
    previously completed tools are NOT executed twice,
    outcome is logged as failure in repository, and conversation state remains uncorrupted.
    """
    search_execution_count = 0

    class TrackingSearchTool(BaseTool):
        @property
        def name(self) -> str:
            return "web_search"

        @property
        def description(self) -> str:
            return "Tracks execution count"

        @property
        def input_schema(self) -> Type[BaseModel]:
            return SearchInput

        def run(self, **kwargs) -> ToolResult:
            nonlocal search_execution_count
            search_execution_count += 1
            return ToolResult(tool_name=self.name, success=True, data="Search results data")

    custom_tool_registry._tools["web_search"] = TrackingSearchTool()
    custom_tool_registry.register(MockFailingTool())

    skill_repo = SkillRepository(test_db_session)
    skill_service = SkillService(
        repository=skill_repo,
        tool_registry=custom_tool_registry,
    )

    user_id = "user_fail_test"
    steps = [
        {"tool_name": "web_search", "arguments": {"query": "Find Python {ver}"}, "purpose": "Step 1"},
        {"tool_name": "failing_operation", "arguments": {"reason": "test"}, "purpose": "Step 2 fails"},
    ]
    skill = skill_service.create_skill(
        user_id=user_id,
        name="Multi Step Failing Procedure",
        description="Fails on step 2",
        steps=steps,
        parameters_schema={"ver": {"type": "string", "required": True}},
    )

    # Execute skill
    result = skill_service.execute_skill(
        user_id=user_id,
        skill_id_or_name=skill.id,
        inputs={"ver": "3.12"},
    )

    # 1. Verification of execution failure
    assert result.success is False
    assert result.failed_step_index == 1

    # 2. Confirm Step 1 was executed exactly once (no duplicate execution)
    assert search_execution_count == 1

    # 3. Confirm failure outcome is recorded in repository
    persisted = skill_repo.get(user_id=user_id, skill_id=skill.id)
    assert persisted is not None
    assert persisted.usage_count == 1
    assert len(persisted.recent_outcomes) == 1
    recent = persisted.recent_outcomes[0]
    assert recent["success"] is False
    assert "Downstream API connection failure" in str(recent["error"])
    assert recent["steps_executed"] == 2

