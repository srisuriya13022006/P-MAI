"""
P6 — Execution Plan Data Structures and Constants.
Represents an explicit, deterministic, inspectable execution plan for agent tools.
"""
from dataclasses import dataclass, field
from typing import Any


MAX_PLAN_STEPS = 4
KNOWN_EXTRACTORS = frozenset({
    "extract_version_comparison",
    "extract_date_offset",
    "extract_search_url",
    "extract_date_for_search",
    "extract_page_calculation",
    "extract_chained_calculation",
})


@dataclass
class ToolStep:
    """A single deterministic tool step in an execution plan."""

    step_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    purpose: str = ""
    extractor: str | None = None  # Identifier for dynamic argument resolution from dependencies


@dataclass
class StepResult:
    """Controlled, normalized result of a single tool step."""

    step_id: str
    tool_name: str
    success: bool
    data: Any = None
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    output_text: str = ""
    skipped: bool = False
    skip_reason: str | None = None

    def to_compact_dict(self) -> dict[str, Any]:
        """Normalize intermediate tool result to a compact structure."""
        return {
            "step_id": self.step_id,
            "tool": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ExecutionPlan:
    """An explicit sequence of tool steps with dependencies and purpose."""

    plan_id: str
    steps: list[ToolStep] = field(default_factory=list)
    final_response_required: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def get_step(self, step_id: str) -> ToolStep | None:
        """Find a step by its step_id."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert ExecutionPlan to a dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "steps": [
                {
                    "step_id": s.step_id,
                    "tool_name": s.tool_name,
                    "arguments": s.arguments,
                    "depends_on": s.depends_on,
                    "purpose": s.purpose,
                    "extractor": s.extractor,
                }
                for s in self.steps
            ],
            "final_response_required": self.final_response_required,
            "context": self.context,
        }
