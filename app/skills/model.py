"""
P16 — Procedural Skill Domain Models & Security Constraints.
Ensures skills remain safe, inspectable data without containing credentials, arbitrary code, or shell commands.
"""
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


class SkillSecurityViolationError(ValueError):
    """Raised when a skill definition violates security boundaries."""
    pass


class SkillValidationError(ValueError):
    """Raised when skill steps or parameters fail structural validation."""
    pass


FORBIDDEN_CREDENTIAL_PATTERNS = [
    re.compile(r"(?:api[_-]?key|bearer\s+[a-zA-Z0-9_\-\.]+|password|secret|token)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]+['\"]", re.IGNORECASE),
    re.compile(r"\b(?:AIzaSy[a-zA-Z0-9_\-]{33}|gsk_[a-zA-Z0-9]{40,})\b"),
    re.compile(r"\b(?:sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36})\b"),
]

FORBIDDEN_EXECUTABLE_PATTERNS = [
    re.compile(r"\b(?:eval|exec|compile|__import__|os\.system|subprocess\.|shutil\.rmtree)\b"),
    re.compile(r"\b(?:bash|sh|cmd\.exe|powershell)\s+-c\b", re.IGNORECASE),
    re.compile(r"\b(?:sudo|rm\s+-rf|chmod\s+\+x|;\s*(?:bash|sh))\b", re.IGNORECASE),
    re.compile(r"^\s*(?:import\s+\w+|from\s+\w+\s+import)\b", re.MULTILINE),
]


def validate_skill_security(
    name: str,
    description: str,
    steps: list[dict[str, Any]],
    parameters_schema: dict[str, Any] | None = None,
) -> None:
    """
    Validate that a proposed skill does NOT contain credentials, tokens, arbitrary Python,
    or shell execution commands.
    """
    corpus_parts = [name, description]
    if parameters_schema:
        corpus_parts.append(str(parameters_schema))

    for step in steps:
        tool_name = step.get("tool_name", "")
        # Tool name must be alphanumeric identifier with underscores
        if not re.match(r"^[a-zA-Z0-9_]+$", tool_name):
            raise SkillSecurityViolationError(f"Invalid tool name in skill: '{tool_name}'")

        purpose = step.get("purpose", "")
        args_str = str(step.get("arguments", {}))
        corpus_parts.extend([tool_name, purpose, args_str])

    combined_text = "\n".join(corpus_parts)

    for pat in FORBIDDEN_CREDENTIAL_PATTERNS:
        if pat.search(combined_text):
            raise SkillSecurityViolationError("Skill definition contains potential credentials, API keys, or tokens.")

    for pat in FORBIDDEN_EXECUTABLE_PATTERNS:
        if pat.search(combined_text):
            raise SkillSecurityViolationError("Skill definition contains unauthorized executable code or shell constructs.")


@dataclass
class SkillProposal:
    """A validated skill proposal presented to the user for explicit confirmation before saving."""
    name: str
    description: str
    intended_use: str
    parameters_schema: dict[str, Any]
    steps: list[dict[str, Any]]
    completion_criteria: str | None = None
    expected_outputs: str | None = None
    verification_criteria: str | None = None
    known_limitations: str | None = None
    provenance: str = "derived_from_task"
    derived_from_task_id: str | None = None

    def to_confirmation_prompt(self) -> str:
        """Format the skill proposal for user inspection and approval."""
        lines = [
            f"I've prepared a reusable skill '{self.name}':",
            f"• Purpose: {self.intended_use or self.description}",
        ]
        if self.parameters_schema:
            param_desc = ", ".join(
                f"{k} ({v.get('type', 'string')}{', required' if v.get('required') else ''})"
                for k, v in self.parameters_schema.items()
            )
            lines.append(f"• Required inputs: {param_desc}")
        else:
            lines.append("• Inputs: None (deterministic)")

        lines.append("• Steps:")
        for idx, step in enumerate(self.steps, 1):
            t_name = step.get("tool_name", "unknown")
            purpose = step.get("purpose", "execute")
            lines.append(f"  {idx}. {purpose} (tool: {t_name})")

        if self.verification_criteria:
            lines.append(f"• Verification: {self.verification_criteria}")
        if self.known_limitations:
            lines.append(f"• Limitations: {self.known_limitations}")

        lines.append("\nWould you like me to save this workflow as a reusable skill?")
        return "\n".join(lines)


@dataclass
class SkillExecutionResult:
    """Summary of an executed skill run."""
    skill_id: str
    skill_name: str
    success: bool
    steps_executed: int
    duration_ms: float
    output_summary: str
    error: str | None = None
    failed_step_index: int | None = None
    intermediate_data: dict[str, Any] = field(default_factory=dict)


# --- Pydantic Schemas for Structured Output Validation (P16.1) ---
from pydantic import BaseModel, Field


class SkillStepModel(BaseModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    purpose: str = ""
    extractor: str | None = None


class SkillProposalModel(BaseModel):
    name: str
    description: str
    intended_use: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    steps: list[SkillStepModel]
    expected_outputs: str | list[str] | None = None
    verification_criteria: str | list[str] | None = None
    known_limitations: str | list[str] | None = None


class AdaptedPlanStepModel(BaseModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    purpose: str = ""
    extractor: str | None = None


class AdaptedPlanModel(BaseModel):
    plan_id: str | None = None
    user_goal: str
    steps: list[AdaptedPlanStepModel]
    explanation: str | None = None


class WorkflowSummaryModel(BaseModel):
    summary: str
    reusable_insights: list[str] = Field(default_factory=list)
    recommended_parameters: list[str] = Field(default_factory=list)

