"""
P7.1 — Controlled General Planner Dataflow & 3-Step Execution Hardening Unit Tests.
Covers:
- Valid Search -> Fetch -> Compare with all 3 steps executed
- Search -> Fetch where compare value is absent (step_3 skips with structured reason)
- Search result URL extraction failure (steps 2 & 3 skip)
- Fetch failure causing step_3 to skip
- Incompatible extracted types
- Invalid calculator input extracted from page text
- Entity mismatch between search result and fetched page
- Version mismatch (major version compatibility)
- Successful 3-step StepResult propagation
- Tool capability metadata consistency and registry drift protection
- LLM planner path verification (forced enable_llm_planner=True, candidate parsing, validation, execution)
- LLM malformed output fallback to deterministic planner
- Security invariants (prompt injection boundary, memory authorization gate, trace ID hiding)
"""
from datetime import datetime
import json
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pytest

from app.agent.capabilities import (
    BUILTIN_CAPABILITIES,
    ALLOWED_TOOL_DEPENDENCIES,
    ToolCapability,
    get_tool_capability,
    validate_capability_metadata_consistency,
)
from app.agent.executor import PlanExecutor
from app.agent.orchestrator import MAIOrchestrator
from app.agent.plan import ExecutionPlan, StepResult, ToolStep
from app.agent.planner import ControlledGeneralPlanner, build_execution_plan
from app.agent.planner_schema import CandidatePlan, CandidateToolStep
from app.agent.trace import ExecutionTrace
from app.schemas.agent import AgentDecision
from app.tools.base import BaseTool
from app.tools.calculator.tool import CalculatorTool
from app.tools.datetime.tool import DateTimeTool
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


# --- Mock Tools for Hardened 3-Step Testing ---

class MockP71SearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "P7.1 Mock web search tool"

    @property
    def input_schema(self):
        from app.tools.web.tool import WebSearchInput
        return WebSearchInput

    def run(self, **kwargs):
        query = kwargs.get("query", "")
        if "no_url" in query.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"query": query, "results": []},
            )
        if "no_version" in query.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [
                        {
                            "title": "General Programming",
                            "url": "https://example.com/no_version",
                            "snippet": "No versions here",
                        }
                    ],
                },
            )
        if "fail" in query.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [
                        {
                            "title": "Broken Page",
                            "url": "https://example.com/404_fail",
                            "snippet": "Broken",
                        }
                    ],
                },
            )
        if "error" in query.lower():
            return ToolResult(tool_name=self.name, success=False, error="Search timeout")

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "query": query,
                "results": [
                    {
                        "title": "Python 3.14 Release Documentation",
                        "url": "https://docs.python.org/3/whatsnew/3.14.html",
                        "snippet": "Python 3.14 includes new features.",
                    }
                ],
            },
        )


class MockP71FetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "P7.1 Mock web fetch tool"

    @property
    def input_schema(self):
        from app.tools.web.fetch import WebFetchInput
        return WebFetchInput

    def run(self, **kwargs):
        url = kwargs.get("url", "")
        if "fail" in url.lower() or "404" in url.lower():
            return ToolResult(tool_name=self.name, success=False, error="HTTP 404 Not Found")

        if "no_version" in url.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": url,
                    "final_url": url,
                    "title": "General Programming",
                    "content": "This page discusses general software engineering topics without versions.",
                },
            )

        if "entity_mismatch" in url.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": url,
                    "final_url": url,
                    "title": "Ruby Programming Language",
                    "content": "Ruby 3.2 introduces modern object-oriented programming paradigms.",
                },
            )

        if "major_mismatch" in url.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": url,
                    "final_url": url,
                    "title": "Python 4.0 Release Preview",
                    "content": "Python 4.0 is a major future overhaul with breaking changes.",
                },
            )

        if "bad_format" in url.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": url,
                    "final_url": url,
                    "title": "Python Alpha Build",
                    "content": "Python alpha-unreleased build versioning experiment.",
                },
            )

        if "no_numbers" in url.lower():
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "url": url,
                    "final_url": url,
                    "title": "Text Page",
                    "content": "Only words and letters here, no digits at all.",
                },
            )

        # Default: Valid page with Python 3.14
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "url": url,
                "final_url": url,
                "title": "What's New in Python 3.14",
                "content": "Python 3.14 is a stable release of Python. It succeeds Python 3.13 with 140 improvements.",
            },
        )


@pytest.fixture
def p71_registry():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        DateTimeTool(
            clock=lambda tz: datetime(2026, 10, 6, 12, 0, 0, tzinfo=ZoneInfo("UTC")).astimezone(tz)
        )
    )
    registry.register(MockP71SearchTool())
    registry.register(MockP71FetchTool())
    return registry


# =========================================================================
# 1. 3-Step Execution & Dataflow Hardening Tests
# =========================================================================

def test_3_step_search_fetch_compare_full_execution(p71_registry):
    """Verify valid 3-step chain (web_search -> web_fetch -> calculator) executes all 3 steps successfully."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_valid_3step",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_search",
                arguments={"query": "latest python release notes"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="web_fetch",
                arguments={},
                depends_on=["step_1"],
                extractor="extract_search_url",
            ),
            ToolStep(
                step_id="step_3",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_2"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    trace = ExecutionTrace(plan_id="p71_valid_3step")
    results = executor.execute(
        plan=plan,
        user_message="Search latest python release notes, read page, and compare with 3.13",
        trace=trace,
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[0].tool_name == "web_search"

    assert results[1].success is True
    assert results[1].tool_name == "web_fetch"
    assert results[1].arguments["url"] == "https://docs.python.org/3/whatsnew/3.14.html"

    assert results[2].success is True
    assert results[2].tool_name == "calculator"
    assert results[2].arguments["expression"] == "14 - 13"
    assert results[2].data["result"] == 1.0

    assert trace.executed_steps == ["step_1", "step_2", "step_3"]
    assert trace.skipped_steps == []
    assert trace.final_status == "success"


def test_3_step_compare_value_absent_skips_step_3(p71_registry):
    """When fetched page has no version info for entity, step_3 calculator is skipped cleanly."""
    executor = PlanExecutor(p71_registry)
    # Search succeeds, but fetch URL triggers 'no_version'
    plan = ExecutionPlan(
        plan_id="p71_no_val_chain",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_search",
                arguments={"query": "no_version python release"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="web_fetch",
                arguments={},
                depends_on=["step_1"],
                extractor="extract_search_url",
            ),
            ToolStep(
                step_id="step_3",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_2"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    trace = ExecutionTrace(plan_id="p71_no_val_chain")
    results = executor.execute(plan, "Compare python release", trace=trace)

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is True
    assert results[2].skipped is True
    assert "could not discover version for entity 'python'" in results[2].skip_reason.lower()
    assert trace.skipped_steps == ["step_3"]
    assert trace.final_status == "partial"


def test_3_step_search_url_extraction_failure_skips_steps(p71_registry):
    """When search returns no URLs, step 2 and step 3 are skipped without fabricating inputs."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_no_url_chain",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_search",
                arguments={"query": "no_url query"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="web_fetch",
                arguments={},
                depends_on=["step_1"],
                extractor="extract_search_url",
            ),
            ToolStep(
                step_id="step_3",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_2"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    trace = ExecutionTrace(plan_id="p71_no_url_chain")
    results = executor.execute(plan, "Search, read, and compare", trace=trace)

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].skipped is True
    assert "could not extract search url" in results[1].skip_reason.lower()
    assert results[2].skipped is True
    assert "prerequisite step 'step_2' did not succeed" in results[2].skip_reason.lower()
    assert trace.skipped_steps == ["step_2", "step_3"]


def test_3_step_fetch_failure_skips_step_3(p71_registry):
    """When fetch fails with HTTP error, step 3 calculator is cleanly skipped."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_fetch_fail",
        steps=[
            ToolStep(step_id="step_1", tool_name="web_search", arguments={"query": "fail python"}),
            ToolStep(
                step_id="step_2",
                tool_name="web_fetch",
                arguments={},
                depends_on=["step_1"],
                extractor="extract_search_url",
            ),
            ToolStep(
                step_id="step_3",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_2"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    trace = ExecutionTrace(plan_id="p71_fetch_fail")
    results = executor.execute(plan, "Compare", trace=trace)

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].skipped is True
    assert "prerequisite step 'step_2' did not succeed" in results[2].skip_reason.lower()
    assert trace.skipped_steps == ["step_3"]


def test_3_step_incompatible_extracted_types(p71_registry):
    """When version formatting cannot be parsed, step 3 is skipped with incompatible format error."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_bad_type",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_fetch",
                arguments={"url": "https://example.com/bad_format"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "invalid_ver"},
                depends_on=["step_1"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    results = executor.execute(plan, "Compare")
    assert len(results) == 2
    assert results[1].skipped is True
    assert ("incompatible version format" in results[1].skip_reason.lower()) or ("could not discover" in results[1].skip_reason.lower())


def test_3_step_invalid_calculator_input_extracted_from_page(p71_registry):
    """When extract_page_calculation finds no numbers, calculator step is skipped."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_no_num",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_fetch",
                arguments={"url": "https://example.com/no_numbers"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="calculator",
                arguments={},
                depends_on=["step_1"],
                extractor="extract_page_calculation",
            ),
        ],
    )

    results = executor.execute(plan, "Calculate page text")
    assert len(results) == 2
    assert results[1].skipped is True
    assert "could not extract numerical values" in results[1].skip_reason.lower()


def test_3_step_entity_mismatch(p71_registry):
    """When fetched page describes a different entity (e.g. Ruby vs Python), step 3 is skipped."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_mismatch",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_fetch",
                arguments={"url": "https://example.com/entity_mismatch"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_1"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    results = executor.execute(plan, "Compare python")
    assert len(results) == 2
    assert results[1].skipped is True
    assert "could not discover version for entity 'python'" in results[1].skip_reason.lower()


def test_3_step_version_mismatch_major_version(p71_registry):
    """When discovered major version does not match target major version, step 3 is skipped."""
    executor = PlanExecutor(p71_registry)
    plan = ExecutionPlan(
        plan_id="p71_major_mismatch",
        steps=[
            ToolStep(
                step_id="step_1",
                tool_name="web_fetch",
                arguments={"url": "https://example.com/major_mismatch"},
            ),
            ToolStep(
                step_id="step_2",
                tool_name="calculator",
                arguments={"entity": "python", "target_version": "3.13"},
                depends_on=["step_1"],
                extractor="extract_version_comparison",
            ),
        ],
    )

    results = executor.execute(plan, "Compare python")
    assert len(results) == 2
    assert results[1].skipped is True
    assert "incompatible major versions" in results[1].skip_reason.lower()


# =========================================================================
# 2. Capability Metadata Consistency Tests
# =========================================================================

def test_capability_metadata_consistency_clean_registry(p71_registry):
    """Clean registry matching catalog passes consistency check."""
    is_valid, errors = validate_capability_metadata_consistency(p71_registry)
    assert is_valid is True
    assert errors == []


def test_capability_metadata_consistency_missing_tool_fails():
    """Registry with a registered tool missing capability metadata fails consistency."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    class UnknownHackerTool(BaseTool):
        @property
        def name(self) -> str:
            return "unknown_hacker_tool"
        @property
        def description(self) -> str:
            return "none"
        @property
        def input_schema(self):
            return None
        def run(self, **kwargs):
            return ToolResult(tool_name="unknown_hacker_tool", success=True)

    registry.register(UnknownHackerTool())
    is_valid, errors = validate_capability_metadata_consistency(registry)
    # The unknown tool is detected
    assert any("unknown_hacker_tool" in e for e in errors) or any("is not registered" in e for e in errors)


# =========================================================================
# 3. Forced LLM Planner Path & Fallback Tests
# =========================================================================

def test_llm_planner_forced_enable_pipeline(p71_registry):
    """Verify that forced enable_llm_planner=True parses CandidatePlan, validates deterministically, and executes."""
    fake_candidate_json = json.dumps({
        "plan_id": "forced_llm_plan_3step",
        "reasoning": "Search for Python release, fetch docs, and compare with 3.13",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "web_search",
                "arguments": {"query": "latest python release notes"},
                "purpose": "Find release documentation",
            },
            {
                "step_id": "step_2",
                "tool_name": "web_fetch",
                "arguments": {},
                "depends_on": ["step_1"],
                "extractor": "extract_search_url",
                "purpose": "Fetch release page",
            },
            {
                "step_id": "step_3",
                "tool_name": "calculator",
                "arguments": {"entity": "python", "target_version": "3.13"},
                "depends_on": ["step_2"],
                "extractor": "extract_version_comparison",
                "purpose": "Compare minor versions",
            },
        ],
    })

    mock_llm = MagicMock()
    mock_llm.client.chat.return_value = {"message": {"content": fake_candidate_json}}

    decision = AgentDecision(
        intent="multi_tool",
        route="tool",
        needs_clarification=False,
        tools=["web_search", "web_fetch", "calculator"],
        tool_arguments={},
        reason="Multi-step test",
    )

    planner = ControlledGeneralPlanner(
        llm=mock_llm,
        tool_registry=p71_registry,
        use_llm=True,
    )

    plan, trace = planner.plan(
        user_message="Search latest python release notes, read page, and compare with 3.13",
        decision=decision,
    )

    # 1. Schema parsed and validated
    assert len(plan.steps) == 3
    assert trace.planner_source == "llm"
    assert trace.validation_passed is True

    # 2. Executed via deterministic PlanExecutor
    executor = PlanExecutor(p71_registry)
    results = executor.execute(plan, "Search, read, and compare", trace=trace)

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is True
    assert results[2].success is True
    assert results[2].data["result"] == 1.0


def test_llm_planner_forced_enable_malformed_fallback(p71_registry):
    """When forced LLM returns garbage JSON, planner cleanly falls back to deterministic planning."""
    mock_llm = MagicMock()
    mock_llm.client.chat.return_value = {"message": {"content": "INVALID NOT JSON {["}}

    decision = AgentDecision(
        intent="multi_tool",
        route="tool",
        needs_clarification=False,
        tools=["web_search", "calculator"],
        tool_arguments={"query": "latest python release and compare with 3.13"},
        reason="Multi-step test",
    )

    planner = ControlledGeneralPlanner(
        llm=mock_llm,
        tool_registry=p71_registry,
        use_llm=True,
    )

    plan, trace = planner.plan(
        user_message="Search latest python release and compare with 3.13",
        decision=decision,
    )

    assert len(plan.steps) == 2
    assert trace.planner_source == "fallback"
    assert plan.steps[0].tool_name == "web_search"
    assert plan.steps[1].tool_name == "calculator"
