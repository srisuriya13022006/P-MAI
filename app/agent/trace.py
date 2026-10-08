"""
P7 — Execution Trace.
Structured observability and debugging trace for plan decomposition, validation, and execution.
"""
from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class ExecutionTrace:
    """Internal structured trace of plan decomposition, validation, and execution."""

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:8]}")
    plan_id: str = ""
    user_message: str = ""
    planner_source: str = "deterministic"  # "llm" | "deterministic" | "fallback"
    candidate_plan: dict[str, Any] | None = None
    validation_passed: bool = False
    validation_error: str | None = None
    executed_steps: list[str] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    final_status: str = "pending"  # "success" | "partial" | "failed" | "skipped" | "validation_failed"
    task_id: str | None = None
    task_lifecycle_before: str | None = None
    task_lifecycle_after: str | None = None
    clarification_requested: bool = False
    clarification_reason: str | None = None
    clarification_id: str | None = None
    clarification_rounds: int = 0
    requested_capabilities: list[str] = field(default_factory=list)
    candidate_tools: list[str] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    selection_reasons: dict[str, Any] = field(default_factory=dict)
    rejected_candidate_reasons: dict[str, Any] = field(default_factory=dict)
    final_validated_plan: dict[str, Any] | None = None
    # P11 — Replanning and Recovery observability
    recovery_triggered: bool = False
    failure_category: str | None = None
    failed_step: str | None = None
    replan_attempt: int = 0
    recovery_reason: str | None = None
    candidate_replan: dict[str, Any] | None = None
    replan_validation_status: str | None = None
    replacement_steps: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    final_recovery_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for inspection and debugging."""
        return {
            "trace_id": self.trace_id,
            "plan_id": self.plan_id,
            "user_message": self.user_message,
            "planner_source": self.planner_source,
            "candidate_plan": self.candidate_plan,
            "validation_passed": self.validation_passed,
            "validation_error": self.validation_error,
            "executed_steps": self.executed_steps,
            "step_results": self.step_results,
            "skipped_steps": self.skipped_steps,
            "failure_reasons": self.failure_reasons,
            "final_status": self.final_status,
            "task_id": self.task_id,
            "task_lifecycle_before": self.task_lifecycle_before,
            "task_lifecycle_after": self.task_lifecycle_after,
            "clarification_requested": self.clarification_requested,
            "clarification_reason": self.clarification_reason,
            "clarification_id": self.clarification_id,
            "clarification_rounds": self.clarification_rounds,
            "requested_capabilities": self.requested_capabilities,
            "candidate_tools": self.candidate_tools,
            "selected_tools": self.selected_tools,
            "selection_reasons": self.selection_reasons,
            "rejected_candidate_reasons": self.rejected_candidate_reasons,
            "final_validated_plan": self.final_validated_plan,
            "recovery_triggered": self.recovery_triggered,
            "failure_category": self.failure_category,
            "failed_step": self.failed_step,
            "replan_attempt": self.replan_attempt,
            "recovery_reason": self.recovery_reason,
            "candidate_replan": self.candidate_replan,
            "replan_validation_status": self.replan_validation_status,
            "replacement_steps": self.replacement_steps,
            "total_tool_calls": self.total_tool_calls,
            "final_recovery_status": self.final_recovery_status,
        }
