"""
P11 — Bounded Replanning, Recovery & Failure-Aware Execution.
Provides structured failure classification, recoverability policies, execution budgets,
plan-difference protection, and validated replacement plan generation.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any
import uuid

from app.agent.capabilities import (
    BUILTIN_CAPABILITIES,
    PlannerCapability,
    ALLOWED_TOOL_DEPENDENCIES,
    get_tool_capability,
)
from app.agent.plan import ExecutionPlan, ToolStep, StepResult, MAX_PLAN_STEPS
from app.agent.task_state import ActiveTaskState, TaskLifecycleStatus
from app.agent.clarification import ClarificationReason, ClarificationRequest
from app.tools.registry import ToolRegistry


# Explicit, bounded execution limits
MAX_REPLAN_ATTEMPTS: int = 2
MAX_TOTAL_TOOL_CALLS: int = 6
MAX_TOTAL_EXECUTION_STEPS: int = 6
MAX_REPLAN_DEPTH: int = 2


class FailureCategory(str, Enum):
    """Explicit taxonomy of tool and execution failures."""

    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    TIMEOUT = "TIMEOUT"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    AUTHORIZATION_FAILURE = "AUTHORIZATION_FAILURE"
    NOT_FOUND = "NOT_FOUND"
    EMPTY_RESULT = "EMPTY_RESULT"
    DATA_EXTRACTION_FAILURE = "DATA_EXTRACTION_FAILURE"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


# Recovery policy mappings
RECOVERABLE_CATEGORIES: set[FailureCategory] = {
    FailureCategory.TIMEOUT,
    FailureCategory.NOT_FOUND,
    FailureCategory.NETWORK_FAILURE,
    FailureCategory.TRANSIENT_FAILURE,
    FailureCategory.DATA_EXTRACTION_FAILURE,
    FailureCategory.EMPTY_RESULT,
}

NON_RECOVERABLE_CATEGORIES: set[FailureCategory] = {
    FailureCategory.SAFETY_VIOLATION,
    FailureCategory.AUTHORIZATION_FAILURE,
    FailureCategory.TOOL_UNAVAILABLE,
    FailureCategory.UNKNOWN_FAILURE,
    FailureCategory.INVALID_INPUT,  # Input invalidity requires clarification or safe failure, never fabricated retry
}


def classify_step_failure(result: StepResult) -> FailureCategory:
    """Classify a failed StepResult into an explicit FailureCategory."""
    err = (result.error or "").lower()
    skip = (result.skip_reason or "").lower()
    combined = f"{err} {skip}".strip()

    # 1. Safety violations (highest priority hard stops, never replan around safety boundaries)
    if any(k in combined for k in ("ssrf", "blocked", "forbidden", "loop protection", "safety_limit", "safety violation", "private network", "loop")):
        return FailureCategory.SAFETY_VIOLATION

    # 2. Authorization / Confirmation failures
    if any(k in combined for k in ("confirmation required", "unauthorized memory", "permission denied")):
        return FailureCategory.AUTHORIZATION_FAILURE

    # 3. Timeouts
    if any(k in combined for k in ("timeout", "timed out", "deadline exceeded")):
        return FailureCategory.TIMEOUT

    # 4. 404 / Page Not Found
    if any(k in combined for k in ("404", "not found", "status 404")):
        return FailureCategory.NOT_FOUND

    # 5. Network / Connection errors
    if any(k in combined for k in ("connection", "network", "unreachable", "dns", "connection refused", "reset by peer")):
        return FailureCategory.NETWORK_FAILURE

    # 6. Extraction failures
    if any(k in combined for k in ("extract", "extraction", "could not extract", "invalid extractor")):
        return FailureCategory.DATA_EXTRACTION_FAILURE

    # 7. Invalid input
    if any(k in combined for k in ("invalid argument", "syntax error", "malformed", "division by zero", "invalid expression")):
        return FailureCategory.INVALID_INPUT

    # 8. Tool unavailable
    if any(k in combined for k in ("not available", "not registered")):
        return FailureCategory.TOOL_UNAVAILABLE

    # 9. Transient server errors
    if any(k in combined for k in ("500", "502", "503", "504", "server error", "gateway timeout", "rate limit", "temporarily unavailable", "transient")):
        return FailureCategory.TRANSIENT_FAILURE

    # 10. Empty result
    if result.success and (result.data == [] or (isinstance(result.data, dict) and not result.data.get("results") and not result.data.get("content"))):
        return FailureCategory.EMPTY_RESULT

    return FailureCategory.UNKNOWN_FAILURE


def is_failure_recoverable(category: FailureCategory) -> bool:
    """Deterministic check if a failure category is permitted to trigger replanning."""
    return category in RECOVERABLE_CATEGORIES


def calculate_step_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Normalized signature for duplicate call / plan-difference protection."""
    filtered_args = {
        k: str(v)
        for k, v in arguments.items()
        if k not in ("user_message", "candidate_pool_size", "confirmed")
    }
    args_json = json.dumps(filtered_args, sort_keys=True)
    return f"{tool_name}:{args_json}"


@dataclass
class ReplanningContext:
    """Context state required to construct a valid, bounded replacement plan."""

    original_user_goal: str
    active_task: ActiveTaskState | None
    completed_steps: list[StepResult]
    failed_step: StepResult
    failure_category: FailureCategory
    attempted_signatures: set[str] = field(default_factory=set)
    replan_count: int = 0
    total_tool_calls: int = 0
    remaining_budget: int = MAX_REPLAN_ATTEMPTS


class ControlledReplanner:
    """
    Controlled Replanner (P11).
    Generates bounded replacement plans for recoverable failures while enforcing
    plan-difference protection, successful-step preservation, typed data contracts,
    and deterministic fallback.
    """

    def __init__(
        self,
        llm: Any = None,
        tool_registry: ToolRegistry | None = None,
        use_llm: bool = False,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.use_llm = use_llm

    def generate_replacement_plan(
        self,
        context: ReplanningContext,
        user_message: str = "",
    ) -> tuple[ExecutionPlan | None, str | None]:
        """
        Produce a candidate replacement plan adhering to all recovery constraints.
        Returns: (replacement_plan, status_or_reason)
        """
        # 1. Budget enforcement
        if context.replan_count >= MAX_REPLAN_ATTEMPTS:
            return None, "replan_limit_exhausted"
        if context.total_tool_calls >= MAX_TOTAL_TOOL_CALLS:
            return None, "tool_call_budget_exhausted"

        # 2. Recoverability check
        if not is_failure_recoverable(context.failure_category):
            return None, f"failure_category_{context.failure_category.value}_not_recoverable"

        # 3. Attempt LLM replacement plan if enabled
        llm_status = None
        if self.use_llm and self.llm is not None:
            llm_plan, llm_status = self._attempt_llm_replanning(context, user_message)
            if llm_plan is not None:
                # Validate plan difference against attempted signatures
                diff_ok = self._verify_plan_difference(llm_plan, context.attempted_signatures)
                if diff_ok:
                    return llm_plan, "llm_replacement"

        # 4. Deterministic fallback recovery strategies
        det_plan, det_reason = self._deterministic_recovery(context, user_message)
        if det_plan is not None:
            diff_ok = self._verify_plan_difference(det_plan, context.attempted_signatures)
            if diff_ok:
                return det_plan, det_reason
            else:
                return None, "repeated_plan_signature_prevented"

        if llm_status and "rejected" in llm_status:
            return None, llm_status

        return None, det_reason or "no_recovery_plan_available"

    def _deterministic_recovery(
        self,
        context: ReplanningContext,
        user_message: str,
    ) -> tuple[ExecutionPlan | None, str | None]:
        """Apply deterministic recovery strategies based on failed tool and task context."""
        failed = context.failed_step
        plan_id = f"replan_{uuid.uuid4().hex[:8]}"

        # Strategy A: web_fetch failure recovery (Alternative URL from existing search results)
        if failed.tool_name == "web_fetch":
            return self._recover_web_fetch(context, plan_id)

        # Strategy B: web_search failure recovery (Alternative query or provider)
        if failed.tool_name == "web_search":
            return self._recover_web_search(context, plan_id)

        # Strategy C: Data extraction failure recovery
        if context.failure_category == FailureCategory.DATA_EXTRACTION_FAILURE:
            return self._recover_extraction_failure(context, plan_id)

        return None, "no_deterministic_strategy"

    def _recover_web_fetch(
        self,
        context: ReplanningContext,
        plan_id: str,
    ) -> tuple[ExecutionPlan | None, str | None]:
        """
        Recover from web_fetch failure by attempting the next safe, unattempted URL
        from existing successful search results. Preserves successful search step.
        """
        candidate_urls: list[str] = []
        if context.active_task:
            candidate_urls = list(context.active_task.context_values.get("candidate_urls", []))

        # Also inspect successful web_search results in completed steps
        if not candidate_urls:
            for cs in context.completed_steps:
                if cs.tool_name == "web_search" and cs.success and cs.data:
                    raw_res = cs.data if isinstance(cs.data, list) else (cs.data.get("results", []) if isinstance(cs.data, dict) else [])
                    for item in raw_res:
                        u = item.get("url") if isinstance(item, dict) else None
                        if u and u not in candidate_urls:
                            candidate_urls.append(u)

        if not candidate_urls:
            return None, "no_alternative_urls_available"

        # Find next unattempted safe URL
        next_url = None
        for u in candidate_urls:
            sig = calculate_step_signature("web_fetch", {"url": u})
            if sig not in context.attempted_signatures:
                next_url = u
                break

        if not next_url:
            return None, "all_candidate_urls_exhausted"

        # Construct replacement step
        replacement_step = ToolStep(
            step_id=f"step_{len(context.completed_steps) + 1}",
            tool_name="web_fetch",
            arguments={"url": next_url},
            depends_on=[],
            purpose=f"Alternative URL fetch: {next_url}",
        )

        steps = [replacement_step]

        # If previous plan had downstream dependent step (e.g. calculator comparison or calculation), re-chain it
        if context.active_task and context.active_task.current_plan:
            orig_plan = context.active_task.current_plan
            for orig_step in orig_plan.steps:
                if context.failed_step.step_id in orig_step.depends_on:
                    downstream_step = ToolStep(
                        step_id=f"step_{len(context.completed_steps) + 2}",
                        tool_name=orig_step.tool_name,
                        arguments=dict(orig_step.arguments),
                        depends_on=[replacement_step.step_id],
                        purpose=orig_step.purpose,
                        extractor=orig_step.extractor,
                    )
                    steps.append(downstream_step)

        return ExecutionPlan(plan_id=plan_id, steps=steps), "alternative_url_selected"

    def _recover_web_search(
        self,
        context: ReplanningContext,
        plan_id: str,
    ) -> tuple[ExecutionPlan | None, str | None]:
        """Recover from web_search timeout or empty results using a simplified query."""
        orig_query = context.failed_step.arguments.get("query", "")
        if not orig_query:
            orig_query = context.original_user_goal

        # Formulate simplified search query
        words = re.findall(r"\b[A-Za-z0-9_\.]+\b", orig_query)
        stop_words = {"the", "a", "an", "for", "in", "and", "or", "of", "to", "what", "is", "tell", "me"}
        filtered = [w for w in words if w.lower() not in stop_words]
        new_query = " ".join(filtered[:6]) if filtered else orig_query

        sig = calculate_step_signature("web_search", {"query": new_query})
        if sig in context.attempted_signatures:
            return None, "search_query_already_attempted"

        step = ToolStep(
            step_id=f"step_{len(context.completed_steps) + 1}",
            tool_name="web_search",
            arguments={"query": new_query},
            depends_on=[],
            purpose=f"Refined web search query: {new_query}",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step]), "search_query_refined"

    def _recover_extraction_failure(
        self,
        context: ReplanningContext,
        plan_id: str,
    ) -> tuple[ExecutionPlan | None, str | None]:
        """Recover from extraction failure by fetching an alternate candidate source page."""
        return self._recover_web_fetch(context, plan_id)

    def _verify_plan_difference(
        self,
        plan: ExecutionPlan,
        attempted_signatures: set[str],
    ) -> bool:
        """Enforce that replacement plan is not an identical repeat of already failed steps."""
        if plan.is_empty:
            return False
        # At least one step in the replacement plan must have an unattempted signature
        for step in plan.steps:
            sig = calculate_step_signature(step.tool_name, step.arguments)
            if sig not in attempted_signatures:
                return True
        return False

    def _attempt_llm_replanning(
        self,
        context: ReplanningContext,
        user_message: str,
    ) -> tuple[ExecutionPlan | None, str]:
        """Prompt LLM for candidate replacement plan and enforce strict schema and capability boundaries."""
        try:
            from app.agent.planner_schema import CandidatePlan
            prompt = (
                f"Original goal: {context.original_user_goal}\n"
                f"Failed step: {context.failed_step.tool_name} with error: {context.failed_step.error}\n"
                f"Propose a valid replacement plan in JSON."
            )
            content = ""
            if hasattr(self.llm, "chat"):
                resp = self.llm.chat(prompt)
                content = resp if isinstance(resp, str) else (resp.get("message", {}).get("content", "") if isinstance(resp, dict) else str(resp))
            elif hasattr(self.llm, "client") and hasattr(self.llm.client, "chat"):
                response = self.llm.client.chat(
                    model=getattr(self.llm, "model", "qwen2.5:3b"),
                    messages=[
                        {"role": "system", "content": "You are the execution recovery planner. Output JSON schema."},
                        {"role": "user", "content": prompt},
                    ],
                    format=CandidatePlan.model_json_schema(),
                )
                content = response["message"]["content"]
            else:
                return None, "llm_unavailable"

            try:
                candidate = CandidatePlan.model_validate_json(content)
            except Exception as parse_e:
                return None, f"malformed_llm_json: {parse_e}"

            # Memory safety: Replanning must NEVER turn a failed action into an unauthorized memory write
            has_memory_step = any(s.tool_name in {"remember_memory", "update_memory", "forget_memory"} for s in candidate.steps)
            if has_memory_step:
                low = (user_message or context.original_user_goal).lower()
                has_explicit_memory = bool(re.search(r"\b(remember|forget|save to memory|update memory|note that)\b", low))
                if not has_explicit_memory:
                    return None, "unauthorized_memory_write_rejected"

            # Capability check: Reject unknown tools immediately
            for s in candidate.steps:
                if s.tool_name not in BUILTIN_CAPABILITIES:
                    return None, f"unknown_tool_rejected: {s.tool_name}"
                if self.tool_registry and not self.tool_registry.has(s.tool_name):
                    return None, f"unregistered_tool_rejected: {s.tool_name}"

            plan_steps = [
                ToolStep(
                    step_id=s.step_id,
                    tool_name=s.tool_name,
                    arguments=s.arguments,
                    depends_on=s.depends_on,
                    purpose=s.purpose,
                    extractor=s.extractor,
                )
                for s in candidate.steps
            ]
            return ExecutionPlan(plan_id=candidate.plan_id, steps=plan_steps), "llm_candidate_generated"
        except Exception as e:
            return None, f"llm_replanning_error: {e}"

    def build_clarification_for_exhausted_recovery(
        self,
        context: ReplanningContext,
    ) -> ClarificationRequest:
        """Construct a precise P9 clarification request when recovery cannot proceed without user input."""
        question = (
            f"I was unable to complete '{context.original_user_goal}' using the available sources. "
            "Could you specify an alternative source URL or provide more details?"
        )
        return ClarificationRequest(
            task_id=context.active_task.task_id if context.active_task else None,
            question=question,
            missing_information="source_specification",
            reason=ClarificationReason.INSUFFICIENT_CONTEXT,
        )
