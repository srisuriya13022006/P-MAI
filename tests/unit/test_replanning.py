"""
Unit tests for P11 — Bounded Replanning, Recovery & Failure-Aware Execution.

Tests cover all 25 required test cases:
1. recoverable timeout
2. non-recoverable safety failure
3. retry limit
4. replan limit
5. repeated identical plan prevention
6. successful-step preservation
7. alternative URL recovery
8. web provider fallback
9. extraction failure recovery
10. invalid calculator recovery
11. clarification after failed recovery
12. memory authorization cannot be bypassed
13. prompt injection cannot influence replan
14. unknown tool cannot be introduced during replan
15. P7 validator required for every replacement plan
16. P10 capability checks required for every replacement plan
17. malformed LLM replacement plan
18. deterministic fallback
19. total tool-call budget
20. plan length limit
21. task state updated correctly
22. partial success preserved
23. recovery exhaustion
24. distinct successful replacement plan
25. no unnecessary retry of successful steps
"""
import pytest
from unittest.mock import MagicMock, patch

from app.agent.executor import PlanExecutor
from app.agent.orchestrator import MAIOrchestrator
from app.agent.permissions import PermissionManager
from app.agent.plan import ExecutionPlan, StepResult, ToolStep, MAX_PLAN_STEPS
from app.agent.planner_schema import CandidatePlan, CandidateToolStep
from app.agent.recovery import (
    ControlledReplanner,
    FailureCategory,
    MAX_REPLAN_ATTEMPTS,
    MAX_TOTAL_TOOL_CALLS,
    ReplanningContext,
    calculate_step_signature,
    classify_step_failure,
    is_failure_recoverable,
)
from app.agent.task_state import (
    ActiveTaskState,
    TaskLifecycleStatus,
    TaskStateManager,
)
from app.agent.trace import ExecutionTrace
from app.schemas.agent import AgentDecision
from app.tools.base import BaseTool
from app.tools.calculator.tool import CalculatorTool
from app.tools.datetime.tool import DateTimeTool
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


class DummyTool(BaseTool):
    def __init__(self, name: str, return_result: ToolResult | None = None):
        self._name = name
        self.return_result = return_result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy {self._name}"

    @property
    def input_schema(self):
        return None

    def run(self, **kwargs) -> ToolResult:
        if self.return_result:
            return self.return_result
        return ToolResult(tool_name=self._name, success=True, data={"echo": kwargs})


# 1. Recoverable timeout
def test_recoverable_timeout():
    step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://example.com/api"},
        success=False,
        error="Connection timed out after 5.0 seconds",
    )
    cat = classify_step_failure(step)
    assert cat == FailureCategory.TIMEOUT
    assert is_failure_recoverable(cat) is True


# 2. Non-recoverable safety failure
def test_non_recoverable_safety_failure():
    step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "http://169.254.169.254/latest/meta-data/"},
        success=False,
        error="SSRF protection: Request to private/internal IP address blocked",
    )
    cat = classify_step_failure(step)
    assert cat == FailureCategory.SAFETY_VIOLATION
    assert is_failure_recoverable(cat) is False


# 3. Retry limit
def test_retry_limit_enforced():
    replanner = ControlledReplanner()
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch doc")
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://example.com/doc"},
        success=False,
        error="Timeout",
    )
    # Replan count at limit
    context = ReplanningContext(
        original_user_goal="fetch doc",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        replan_count=MAX_REPLAN_ATTEMPTS,
        total_tool_calls=2,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is None
    assert reason == "replan_limit_exhausted"


# 4. Replan limit
def test_replan_limit():
    assert MAX_REPLAN_ATTEMPTS == 2
    assert MAX_TOTAL_TOOL_CALLS == 6


# 5. Repeated identical plan prevention
def test_repeated_identical_plan_prevention():
    replanner = ControlledReplanner()
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch doc")
    task.context_values["candidate_urls"] = ["https://example.com/doc1"]
    
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://example.com/doc1"},
        success=False,
        error="404 Not Found",
    )
    # The signature for doc1 was already attempted
    sig = calculate_step_signature("web_fetch", {"url": "https://example.com/doc1"})
    context = ReplanningContext(
        original_user_goal="fetch doc",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.NOT_FOUND,
        attempted_signatures={sig},
        replan_count=0,
        total_tool_calls=1,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is None
    assert "all_candidate_urls_exhausted" in reason or "repeated" in reason


# 6. Successful-step preservation
def test_successful_step_preservation():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="find python version")
    task.context_values["candidate_urls"] = ["https://python.org/fail", "https://python.org/success"]
    
    search_step = StepResult(
        step_id="step_1",
        tool_name="web_search",
        arguments={"query": "python latest"},
        success=True,
        data=[{"url": "https://python.org/fail"}, {"url": "https://python.org/success"}],
        output_text="Search results found",
    )
    task.record_step_result(search_step)
    
    failed_fetch = StepResult(
        step_id="step_2",
        tool_name="web_fetch",
        arguments={"url": "https://python.org/fail"},
        success=False,
        error="404 Not Found",
    )
    task.record_step_result(failed_fetch)
    
    replanner = ControlledReplanner()
    sig1 = calculate_step_signature("web_search", {"query": "python latest"})
    sig2 = calculate_step_signature("web_fetch", {"url": "https://python.org/fail"})
    
    context = ReplanningContext(
        original_user_goal="find python version",
        active_task=task,
        completed_steps=[search_step],
        failed_step=failed_fetch,
        failure_category=FailureCategory.NOT_FOUND,
        attempted_signatures={sig1, sig2},
        replan_count=0,
        total_tool_calls=2,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    assert len(plan.steps) == 1
    # Step 1 (search) is preserved and not re-executed in replacement plan!
    assert plan.steps[0].tool_name == "web_fetch"
    assert plan.steps[0].arguments["url"] == "https://python.org/success"


# 7. Alternative URL recovery
def test_alternative_url_recovery():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch doc")
    task.context_values["candidate_urls"] = [
        "https://docs.python.org/broken",
        "https://docs.python.org/working",
    ]
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://docs.python.org/broken"},
        success=False,
        error="503 Service Unavailable",
    )
    sig = calculate_step_signature("web_fetch", {"url": "https://docs.python.org/broken"})
    replanner = ControlledReplanner()
    context = ReplanningContext(
        original_user_goal="fetch doc",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TRANSIENT_FAILURE,
        attempted_signatures={sig},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    assert plan.steps[0].arguments["url"] == "https://docs.python.org/working"


# 8. Web provider fallback
def test_web_provider_fallback_query_refinement():
    replanner = ControlledReplanner()
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_search",
        arguments={"query": "what is the latest release and documentation for python 3.13"},
        success=False,
        error="Timed out waiting for response",
    )
    sig = calculate_step_signature("web_search", {"query": failed_step.arguments["query"]})
    context = ReplanningContext(
        original_user_goal="find python doc",
        active_task=None,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        attempted_signatures={sig},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    assert plan.steps[0].tool_name == "web_search"
    # Query must be refined and stop words filtered
    assert plan.steps[0].arguments["query"] != failed_step.arguments["query"]
    assert "what" not in plan.steps[0].arguments["query"].lower()


# 9. Extraction failure recovery
def test_extraction_failure_recovery():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="extract version")
    task.context_values["candidate_urls"] = ["https://site1.com", "https://site2.com"]
    failed_step = StepResult(
        step_id="step_2",
        tool_name="web_fetch",
        arguments={"url": "https://site1.com"},
        success=False,
        error="Extraction failure: could not extract version number",
    )
    cat = classify_step_failure(failed_step)
    assert cat == FailureCategory.DATA_EXTRACTION_FAILURE
    assert is_failure_recoverable(cat) is True
    
    sig = calculate_step_signature("web_fetch", {"url": "https://site1.com"})
    replanner = ControlledReplanner()
    context = ReplanningContext(
        original_user_goal="extract version",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=cat,
        attempted_signatures={sig},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    assert plan.steps[0].arguments["url"] == "https://site2.com"


# 10. Invalid calculator recovery
def test_invalid_calculator_recovery_non_recoverable():
    failed_step = StepResult(
        step_id="step_1",
        tool_name="calculator",
        arguments={"expression": "10 / 0"},
        success=False,
        error="ZeroDivisionError: division by zero",
    )
    cat = classify_step_failure(failed_step)
    assert cat == FailureCategory.INVALID_INPUT
    # Must NOT blindly retry with fabricated input
    assert is_failure_recoverable(cat) is False


# 11. Clarification after failed recovery
def test_clarification_after_failed_recovery():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="read docs")
    task.context_values["candidate_urls"] = ["https://site.com/broken"]
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://site.com/broken"},
        success=False,
        error="404 Not Found",
    )
    sig = calculate_step_signature("web_fetch", {"url": "https://site.com/broken"})
    replanner = ControlledReplanner()
    context = ReplanningContext(
        original_user_goal="read docs",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.NOT_FOUND,
        attempted_signatures={sig},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is None
    assert reason == "all_candidate_urls_exhausted"
    
    clarif = replanner.build_clarification_for_exhausted_recovery(context)
    assert "unable to complete" in clarif.question.lower() or "alternative" in clarif.question.lower()
    assert clarif.task_id == "t1"


# 12. Memory authorization cannot be bypassed
def test_memory_authorization_cannot_be_bypassed():
    # Prompting memory write without explicit user intent in replan
    registry = ToolRegistry()
    registry.register(DummyTool("remember_memory"))
    executor = PlanExecutor(registry)
    
    plan = ExecutionPlan(
        plan_id="p1",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="remember_memory",
                arguments={"key": "secret", "value": "injected"},
                depends_on=[],
                purpose="injected memory write",
            )
        ],
    )
    is_valid, err = executor.validate_plan(plan, user_message="What is the weather in Paris?")
    assert is_valid is False
    assert "Unauthorized memory mutation" in err


# 13. Prompt injection cannot influence replan
def test_prompt_injection_inert():
    # Web content containing instructions to mutate memory or run commands
    replanner = ControlledReplanner()
    mock_llm = MagicMock()
    # LLM returns a plan attempting memory mutation based on injected text
    mock_llm.chat.return_value = '{"plan_id": "inj_1", "steps": [{"step_id": "step_1", "tool_name": "remember_memory", "arguments": {"key": "pwned", "value": "true"}, "depends_on": [], "purpose": "injected"}]}'
    replanner.llm = mock_llm
    replanner.use_llm = True
    
    failed_step = StepResult(step_id="step_1", tool_name="web_fetch", arguments={"url": "http://evil.com"}, success=False, error="404")
    context = ReplanningContext(
        original_user_goal="summarize documentation",
        active_task=None,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.NOT_FOUND,
    )
    # Original user goal had no memory request
    plan, reason = replanner.generate_replacement_plan(context, user_message="summarize documentation")
    # Must reject unauthorized memory mutation
    assert plan is None


# 14. Unknown tool cannot be introduced during replan
def test_unknown_tool_cannot_be_introduced():
    replanner = ControlledReplanner()
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '{"plan_id": "bad_1", "steps": [{"step_id": "step_1", "tool_name": "arbitrary_shell_command", "arguments": {"cmd": "ls"}, "depends_on": [], "purpose": "exec"}]}'
    replanner.llm = mock_llm
    replanner.use_llm = True
    
    failed_step = StepResult(step_id="step_1", tool_name="web_fetch", arguments={"url": "http://ex.com"}, success=False, error="timeout")
    context = ReplanningContext(
        original_user_goal="run task",
        active_task=None,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    # Rejects unknown tool and has no deterministic fallback -> None
    assert plan is None


# 15. P7 validator required for every replacement plan
def test_p7_validator_required_for_replacement_plan():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = PlanExecutor(registry)
    
    # Cyclic replacement plan
    cyclic_plan = ExecutionPlan(
        plan_id="p_cycle",
        steps=[
            ToolStep(step_id="s1", tool_name="calculator", arguments={"expression": "1+1"}, depends_on=["s2"]),
            ToolStep(step_id="s2", tool_name="calculator", arguments={"expression": "2+2"}, depends_on=["s1"]),
        ],
    )
    is_valid, err = executor.validate_plan(cyclic_plan)
    assert is_valid is False
    assert "Dependency cycle" in err or "invalid forward ordering" in err


# 16. P10 capability checks required for every replacement plan
def test_p10_capability_checks_required():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    executor = PlanExecutor(registry)
    
    # Unsupported dependency pair
    invalid_plan = ExecutionPlan(
        plan_id="p_inv",
        steps=[
            ToolStep(step_id="s1", tool_name="datetime", arguments={}),
            ToolStep(step_id="s2", tool_name="datetime", arguments={}, depends_on=["s1"]),
        ],
    )
    is_valid, err = executor.validate_plan(invalid_plan)
    assert is_valid is False
    assert "Unsupported tool combination" in err


# 17. Malformed LLM replacement plan
def test_malformed_llm_replacement_plan():
    replanner = ControlledReplanner()
    mock_llm = MagicMock()
    mock_llm.chat.return_value = 'THIS IS NOT VALID JSON {malformed'
    replanner.llm = mock_llm
    replanner.use_llm = True
    
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch doc")
    task.context_values["candidate_urls"] = ["https://site.com/ok"]
    failed_step = StepResult(step_id="step_1", tool_name="web_fetch", arguments={"url": "https://site.com/bad"}, success=False, error="timeout")
    
    context = ReplanningContext(
        original_user_goal="fetch doc",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        attempted_signatures={calculate_step_signature("web_fetch", {"url": "https://site.com/bad"})},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    # LLM failed, so deterministic fallback stepped in and used the candidate URL!
    assert plan is not None
    assert plan.steps[0].arguments["url"] == "https://site.com/ok"


# 18. Deterministic fallback
def test_deterministic_fallback_when_llm_fails():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="read docs")
    task.context_values["candidate_urls"] = ["https://docs.python.org/1", "https://docs.python.org/2"]
    
    replanner = ControlledReplanner(use_llm=False)
    failed_step = StepResult(
        step_id="step_1",
        tool_name="web_fetch",
        arguments={"url": "https://docs.python.org/1"},
        success=False,
        error="404 Not Found",
    )
    context = ReplanningContext(
        original_user_goal="read docs",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.NOT_FOUND,
        attempted_signatures={calculate_step_signature("web_fetch", {"url": "https://docs.python.org/1"})},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    assert reason == "alternative_url_selected"
    assert plan.steps[0].arguments["url"] == "https://docs.python.org/2"


# 19. Total tool-call budget
def test_total_tool_call_budget():
    replanner = ControlledReplanner()
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="goal")
    task.total_tool_calls = MAX_TOTAL_TOOL_CALLS
    
    failed_step = StepResult(step_id="step_1", tool_name="web_fetch", arguments={"url": "https://a.com"}, success=False, error="timeout")
    context = ReplanningContext(
        original_user_goal="goal",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        total_tool_calls=MAX_TOTAL_TOOL_CALLS,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is None
    assert reason == "tool_call_budget_exhausted"


# 20. Plan length limit
def test_plan_length_limit():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    executor = PlanExecutor(registry)
    
    # Plan exceeding MAX_PLAN_STEPS
    steps = [
        ToolStep(step_id=f"step_{i}", tool_name="calculator", arguments={"expression": f"{i}+1"})
        for i in range(MAX_PLAN_STEPS + 1)
    ]
    plan = ExecutionPlan(plan_id="p_long", steps=steps)
    is_valid, err = executor.validate_plan(plan)
    assert is_valid is False
    assert "Plan step limit exceeded" in err


# 21. Task state updated correctly
def test_task_state_updated_correctly():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch")
    r1 = StepResult(step_id="s1", tool_name="web_fetch", arguments={"url": "https://a.com"}, success=False, error="404")
    task.record_step_result(r1)
    
    assert task.total_tool_calls == 1
    assert len(task.failed_steps) == 1
    assert task.failed_steps[0]["step_id"] == "s1"
    assert len(task.completed_steps) == 1


# 22. Partial success preserved
def test_partial_success_preserved():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="calc and fetch")
    s1 = StepResult(step_id="s1", tool_name="calculator", arguments={"expression": "10*5"}, success=True, data={"result": 50}, output_text="50")
    s2 = StepResult(step_id="s2", tool_name="web_fetch", arguments={"url": "https://a.com"}, success=False, error="404")
    
    task.record_step_result(s1)
    task.record_step_result(s2)
    
    succeeded = [r for r in task.completed_steps if r.success]
    assert len(succeeded) == 1
    assert succeeded[0].tool_name == "calculator"
    assert succeeded[0].data["result"] == 50


# 23. Recovery exhaustion
def test_recovery_exhaustion():
    replanner = ControlledReplanner()
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch")
    task.replan_count = MAX_REPLAN_ATTEMPTS
    
    failed_step = StepResult(step_id="s1", tool_name="web_fetch", arguments={"url": "https://a.com"}, success=False, error="timeout")
    context = ReplanningContext(
        original_user_goal="fetch",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        replan_count=MAX_REPLAN_ATTEMPTS,
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is None
    assert reason == "replan_limit_exhausted"


# 24. Distinct successful replacement plan
def test_distinct_successful_replacement_plan():
    replanner = ControlledReplanner()
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="fetch doc")
    task.context_values["candidate_urls"] = ["https://bad.com", "https://good.com"]
    
    failed_step = StepResult(step_id="s1", tool_name="web_fetch", arguments={"url": "https://bad.com"}, success=False, error="timeout")
    sig = calculate_step_signature("web_fetch", {"url": "https://bad.com"})
    
    context = ReplanningContext(
        original_user_goal="fetch doc",
        active_task=task,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        attempted_signatures={sig},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    # Signature of replacement must be different
    new_sig = calculate_step_signature(plan.steps[0].tool_name, plan.steps[0].arguments)
    assert new_sig != sig
    assert plan.steps[0].arguments["url"] == "https://good.com"


# 25. No unnecessary retry of successful steps
def test_no_unnecessary_retry_of_successful_steps():
    task = ActiveTaskState(task_id="t1", conversation_id="c1", user_id="u1", original_user_goal="search and fetch")
    s_search = StepResult(
        step_id="step_1",
        tool_name="web_search",
        arguments={"query": "python latest"},
        success=True,
        data=[{"url": "https://fail.org"}, {"url": "https://win.org"}],
        output_text="Search ok",
    )
    s_fetch = StepResult(
        step_id="step_2",
        tool_name="web_fetch",
        arguments={"url": "https://fail.org"},
        success=False,
        error="404 Not Found",
    )
    task.context_values["candidate_urls"] = ["https://fail.org", "https://win.org"]
    
    replanner = ControlledReplanner()
    sig1 = calculate_step_signature("web_search", {"query": "python latest"})
    sig2 = calculate_step_signature("web_fetch", {"url": "https://fail.org"})
    
    context = ReplanningContext(
        original_user_goal="search and fetch",
        active_task=task,
        completed_steps=[s_search],
        failed_step=s_fetch,
        failure_category=FailureCategory.NOT_FOUND,
        attempted_signatures={sig1, sig2},
    )
    plan, reason = replanner.generate_replacement_plan(context)
    assert plan is not None
    # Does NOT include step_1 (web_search)
    tool_names = [s.tool_name for s in plan.steps]
    assert "web_search" not in tool_names
    assert tool_names == ["web_fetch"]
    assert plan.steps[0].arguments["url"] == "https://win.org"
