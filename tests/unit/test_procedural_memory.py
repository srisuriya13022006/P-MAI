"""
Tests for P16: Local-First Procedural Memory & Reusable Skills.
Covers all 12 scenarios required by P16 Section 8.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, Skill
from app.database.repositories.skills import SkillRepository
from app.skills.service import SkillService
from app.skills.model import (
    SkillProposal,
    SkillExecutionResult,
    SkillValidationError,
    SkillSecurityViolationError,
    validate_skill_security,
)
from app.agent.task_state import TaskStateManager, TaskLifecycleStatus
from app.agent.orchestrator import MAIOrchestrator
from app.tools.registry import ToolRegistry
from app.tools.base import BaseTool, ToolResult


from pydantic import BaseModel
from typing import Type

class DummySchema(BaseModel):
    model_config = {"extra": "allow"}


class MockSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web"

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummySchema

    def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        return ToolResult(tool_name=self.name, success=True, data=f"Search results for: {query}")


class MockFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetches a URL"

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummySchema

    def run(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        return ToolResult(tool_name=self.name, success=True, data=f"Fetched content from: {url}")


class MockFailingTool(BaseTool):
    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "Always fails"

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummySchema

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, success=False, error="Simulated failure in tool")


@pytest.fixture
def db_session():
    """Provides a fresh in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def tool_registry():
    """Provides a tool registry with search and fetch tools."""
    registry = ToolRegistry()
    registry.register(MockSearchTool())
    registry.register(MockFetchTool())
    return registry


@pytest.fixture
def skill_service(db_session, tool_registry):
    """Provides a SkillService with in-memory SQLite repository."""
    repo = SkillRepository(db_session)
    return SkillService(repository=repo, tool_registry=tool_registry)


# Scenario 1: Successfully saving an approved skill
def test_1_successfully_save_approved_skill(db_session, skill_service):
    user_id = "user_alpha"
    steps = [
        {"tool_name": "web_search", "arguments": {"query": "{topic}"}, "purpose": "Search topic"},
        {"tool_name": "web_fetch", "arguments": {"url": "https://example.com"}, "purpose": "Fetch source"},
    ]
    schema = {"topic": {"type": "string", "required": True}}

    skill = skill_service.create_skill(
        user_id=user_id,
        name="Research Brief Builder",
        description="Gathers research and fetches sources for a topic",
        steps=steps,
        parameters_schema=schema,
        intended_use="Generate structured briefs",
    )

    assert skill.id is not None
    assert skill.name == "Research Brief Builder"
    assert skill.version == 1
    assert skill.is_active is True
    assert len(skill.steps) == 2
    assert "topic" in skill.parameters_schema

    # Verify persisted in database
    retrieved = skill_service.get_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert retrieved is not None
    assert retrieved.name == "Research Brief Builder"


# Scenario 2: Declining a skill-save proposal produces no new skill
def test_2_declining_skill_proposal_produces_no_skill(skill_service):
    user_id = "user_beta"
    state_mgr = TaskStateManager()
    task = state_mgr.create_task(user_id=user_id, conversation_id="conv_1", goal="Test workflow")
    task.completed_steps = [
        {"tool_name": "web_search", "arguments": {"query": "pytest"}, "result": "done"}
    ]

    proposal = skill_service.propose_from_task(task, name="Proposed Search")
    assert proposal.name == "Proposed Search"

    # Simulate decline
    task.pending_skill_save = None
    task.transition_to(TaskLifecycleStatus.CANCELLED)

    skills = skill_service.list_skills(user_id=user_id)
    assert len(skills) == 0


# Scenario 3: Listing and retrieving saved skills across conversations
def test_3_list_and_retrieve_across_conversations(skill_service):
    user_id = "user_gamma"
    steps = [{"tool_name": "web_search", "arguments": {"query": "{q}"}, "purpose": "Search"}]

    skill1 = skill_service.create_skill(
        user_id=user_id,
        name="Topic Finder",
        description="Finds articles on topic",
        steps=steps,
        parameters_schema={"q": {"type": "string"}},
    )
    skill2 = skill_service.create_skill(
        user_id=user_id,
        name="URL Reader",
        description="Reads webpage content",
        steps=[{"tool_name": "web_fetch", "arguments": {"url": "{url}"}, "purpose": "Read"}],
        parameters_schema={"url": {"type": "string"}},
    )

    # List across conversations
    all_skills = skill_service.list_skills(user_id=user_id)
    assert len(all_skills) == 2
    names = {s.name for s in all_skills}
    assert "Topic Finder" in names
    assert "URL Reader" in names

    # Retrieve by name
    fetched = skill_service.get_skill(user_id=user_id, skill_id_or_name="Topic Finder")
    assert fetched is not None
    assert fetched.id == skill1.id


# Scenario 4: Reusing a skill with different valid input parameters
def test_4_reusing_skill_with_different_valid_inputs(skill_service):
    user_id = "user_delta"
    steps = [{"tool_name": "web_search", "arguments": {"query": "Learn {language}"}, "purpose": "Search tutorial"}]
    skill = skill_service.create_skill(
        user_id=user_id,
        name="Language Tutorial Finder",
        description="Finds tutorials for languages",
        steps=steps,
        parameters_schema={"language": {"type": "string", "required": True}},
    )

    # Run with Python
    res1 = skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill.id, inputs={"language": "Python"})
    assert res1.success is True
    assert "Search results for: Learn Python" in res1.output_summary

    # Run with Rust
    res2 = skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill.id, inputs={"language": "Rust"})
    assert res2.success is True
    assert "Search results for: Learn Rust" in res2.output_summary

    # Verify usage count incremented
    updated = skill_service.get_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert updated.usage_count == 2


# Scenario 5: Rejecting invalid or incomplete inputs
def test_5_rejecting_invalid_or_incomplete_inputs(skill_service):
    user_id = "user_epsilon"
    steps = [{"tool_name": "web_search", "arguments": {"query": "{query}"}, "purpose": "Search"}]
    skill = skill_service.create_skill(
        user_id=user_id,
        name="Mandatory Search",
        description="Requires query",
        steps=steps,
        parameters_schema={"query": {"type": "string", "required": True}},
    )

    # Missing required parameter
    with pytest.raises(SkillValidationError) as exc_info:
        skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill.id, inputs={})

    assert "Missing required parameter 'query'" in str(exc_info.value)


# Scenario 6: Preventing cross-user skill access
def test_6_preventing_cross_user_skill_access(skill_service):
    user_a = "alice"
    user_b = "bob"

    skill_a = skill_service.create_skill(
        user_id=user_a,
        name="Alice Private Skill",
        description="Only Alice can use this",
        steps=[{"tool_name": "web_search", "arguments": {"query": "secret"}, "purpose": "Search"}],
    )

    # Bob lists skills -> should be empty
    bob_skills = skill_service.list_skills(user_id=user_b)
    assert len(bob_skills) == 0

    # Bob retrieves Alice's skill -> None
    retrieved_by_bob = skill_service.get_skill(user_id=user_b, skill_id_or_name=skill_a.id)
    assert retrieved_by_bob is None

    # Bob executes Alice's skill -> ValueError
    with pytest.raises(ValueError) as exc:
        skill_service.execute_skill(user_id=user_b, skill_id_or_name=skill_a.id, inputs={})
    assert "Skill not found" in str(exc.value)

    # Bob disables Alice's skill -> False
    assert skill_service.disable_skill(user_id=user_b, skill_id_or_name=skill_a.id) is False


# Scenario 7: Validating a saved procedure against the current tool registry
def test_7_validate_procedure_against_tool_registry(skill_service):
    user_id = "user_zeta"
    # Procedure references a non-existent tool
    steps = [{"tool_name": "non_existent_tool", "arguments": {"cmd": "run"}, "purpose": "Do something"}]

    validation = skill_service.validate_skill_definition(
        user_id=user_id,
        name="Invalid Tool Skill",
        description="Uses non-existent tool",
        steps=steps,
    )
    assert validation["valid"] is False
    assert any("Tool 'non_existent_tool' is not registered" in err for err in validation["errors"])


# Scenario 8: Enforcing existing plan limits and security checks
def test_8_enforcing_security_and_plan_limits(skill_service):
    user_id = "user_eta"

    # Shell injection / command execution attempt
    insecure_steps = [
        {"tool_name": "web_search", "arguments": {"query": "test; rm -rf /; sudo bash"}, "purpose": "Attack"}
    ]
    with pytest.raises(SkillSecurityViolationError):
        skill_service.create_skill(
            user_id=user_id,
            name="Malicious Skill",
            description="Tries command injection",
            steps=insecure_steps,
        )

    # Hardcoded credential / secret pattern
    secret_steps = [
        {"tool_name": "web_search", "arguments": {"query": "sk-1234567890abcdef1234567890abcdef"}, "purpose": "Leak"}
    ]
    with pytest.raises(SkillSecurityViolationError):
        skill_service.create_skill(
            user_id=user_id,
            name="Secret Leaking Skill",
            description="Contains API token",
            steps=secret_steps,
        )

    # Step limit enforcement (exceeding max 10 steps)
    too_many_steps = [
        {"tool_name": "web_search", "arguments": {"query": f"step_{i}"}, "purpose": f"Step {i}"}
        for i in range(12)
    ]
    with pytest.raises(SkillValidationError):
        skill_service.create_skill(
            user_id=user_id,
            name="Bloated Skill",
            description="Exceeds maximum step limit",
            steps=too_many_steps,
        )


# Scenario 9: Preventing duplicate tool execution during failure recovery
def test_9_preventing_duplicate_tool_execution_on_failure(skill_service, tool_registry):
    tool_registry._tools["failing_tool"] = MockFailingTool()
    user_id = "user_theta"

    search_call_count = 0

    class CountingSearchTool(BaseTool):
        @property
        def name(self) -> str:
            return "web_search"

        @property
        def description(self) -> str:
            return "Counts executions"

        @property
        def input_schema(self) -> Type[BaseModel]:
            return DummySchema

        def run(self, **kwargs) -> ToolResult:
            nonlocal search_call_count
            search_call_count += 1
            query = kwargs.get("query", "")
            return ToolResult(tool_name=self.name, success=True, data=f"Searched {query}")

    tool_registry._tools["web_search"] = CountingSearchTool()

    steps = [
        {"tool_name": "web_search", "arguments": {"query": "initial"}, "purpose": "Step 1"},
        {"tool_name": "failing_tool", "arguments": {}, "purpose": "Step 2 fails"},
    ]

    skill = skill_service.create_skill(
        user_id=user_id,
        name="Partial Failing Skill",
        description="Fails at step 2",
        steps=steps,
    )

    result = skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert result.success is False
    assert result.failed_step_index == 1
    # Step 1 was executed exactly once
    assert search_call_count == 1


# Scenario 10: Recording success and failure outcomes
def test_10_recording_success_and_failure_outcomes(skill_service, tool_registry):
    tool_registry._tools["failing_tool"] = MockFailingTool()
    user_id = "user_iota"

    skill_ok = skill_service.create_skill(
        user_id=user_id,
        name="Healthy Skill",
        description="Always succeeds",
        steps=[{"tool_name": "web_search", "arguments": {"query": "ok"}, "purpose": "Search"}],
    )

    res_ok = skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill_ok.id)
    assert res_ok.success is True

    retrieved_ok = skill_service.get_skill(user_id=user_id, skill_id_or_name=skill_ok.id)
    assert retrieved_ok.usage_count == 1
    assert len(retrieved_ok.recent_outcomes) == 1
    assert retrieved_ok.recent_outcomes[0]["success"] is True
    assert retrieved_ok.recent_outcomes[0]["steps_executed"] == 1
    assert retrieved_ok.recent_outcomes[0]["duration_ms"] >= 0

    # Test failure outcome
    skill_fail = skill_service.create_skill(
        user_id=user_id,
        name="Unhealthy Skill",
        description="Fails immediately",
        steps=[{"tool_name": "failing_tool", "arguments": {}, "purpose": "Fail"}],
    )

    res_fail = skill_service.execute_skill(user_id=user_id, skill_id_or_name=skill_fail.id)
    assert res_fail.success is False

    retrieved_fail = skill_service.get_skill(user_id=user_id, skill_id_or_name=skill_fail.id)
    assert retrieved_fail.usage_count == 1
    assert len(retrieved_fail.recent_outcomes) == 1
    assert retrieved_fail.recent_outcomes[0]["success"] is False
    assert "Simulated failure" in retrieved_fail.recent_outcomes[0]["error"]


# Scenario 11: Disabling, updating, and deleting a skill safely
def test_11_disabling_updating_deleting_skill(skill_service):
    user_id = "user_kappa"

    skill = skill_service.create_skill(
        user_id=user_id,
        name="Lifecycle Skill",
        description="Test lifecycle operations",
        steps=[{"tool_name": "web_search", "arguments": {"query": "v1"}, "purpose": "Step v1"}],
    )
    assert skill.version == 1

    # Update description and steps -> increments version
    updated = skill_service.update_skill(
        user_id=user_id,
        skill_id_or_name=skill.id,
        description="Updated v2 description",
        steps=[{"tool_name": "web_search", "arguments": {"query": "v2"}, "purpose": "Step v2"}],
    )
    assert updated.version == 2
    assert updated.description == "Updated v2 description"

    # Disable skill
    disabled = skill_service.disable_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert disabled is True

    # Active-only list should not include disabled skill
    active_skills = skill_service.list_skills(user_id=user_id, active_only=True)
    assert len(active_skills) == 0

    # Search should ignore disabled skill
    search_res = skill_service.search_skills(user_id=user_id, query="Lifecycle")
    assert search_res.selected_skill is None

    # Reactivate skill
    reactivated = skill_service.reactivate_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert reactivated is True
    active_again = skill_service.list_skills(user_id=user_id, active_only=True)
    assert len(active_again) == 1

    # Delete skill
    deleted = skill_service.delete_skill(user_id=user_id, skill_id_or_name=skill.id)
    assert deleted is True

    # Should no longer exist
    assert skill_service.get_skill(user_id=user_id, skill_id_or_name=skill.id) is None


# Scenario 12: Handling ambiguous skill selection without executing the wrong workflow
def test_12_handling_ambiguous_skill_selection(skill_service):
    user_id = "user_lambda"

    # Create two closely matching skills
    skill1 = skill_service.create_skill(
        user_id=user_id,
        name="Competitor Research Report",
        description="Research market competitors and produce report",
        steps=[{"tool_name": "web_search", "arguments": {"query": "{comp} competitors"}, "purpose": "Search"}],
        parameters_schema={"comp": {"type": "string"}},
    )
    skill2 = skill_service.create_skill(
        user_id=user_id,
        name="Industry Research Report",
        description="Research industry trends and produce report",
        steps=[{"tool_name": "web_search", "arguments": {"query": "{ind} trends"}, "purpose": "Search"}],
        parameters_schema={"ind": {"type": "string"}},
    )

    # Ambiguous query
    outcome = skill_service.search_skills(user_id=user_id, query="Research Report")
    assert outcome.is_ambiguous is True
    assert outcome.selected_skill is None
    candidate_ids = {s.id for s in outcome.ambiguous_candidates}
    assert skill1.id in candidate_ids
    assert skill2.id in candidate_ids

    # Unambiguous specific query
    specific_outcome = skill_service.search_skills(user_id=user_id, query="Competitor Research")
    assert specific_outcome.is_ambiguous is False
    assert specific_outcome.selected_skill is not None
    assert specific_outcome.selected_skill.id == skill1.id
