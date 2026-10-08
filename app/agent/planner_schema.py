"""
P7 — Controlled Planner Schema.
Strict Pydantic models for structured candidate execution plans.
"""
from typing import Any
from pydantic import BaseModel, Field, field_validator


class CandidateToolStep(BaseModel):
    """Schema for a single candidate tool step."""

    step_id: str = Field(..., description="Unique step identifier, e.g. 'step_1'")
    tool_name: str = Field(..., description="Registered tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Static arguments for tool")
    depends_on: list[str] = Field(default_factory=list, description="Step IDs this step depends on")
    extractor: str | None = Field(default=None, description="Optional intermediate data extractor")
    purpose: str = Field(default="", description="Human-readable purpose of this step")

    @field_validator("step_id")
    @classmethod
    def validate_step_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("step_id cannot be empty")
        return v.strip()

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tool_name cannot be empty")
        return v.strip().lower()


class CandidatePlan(BaseModel):
    """Schema for the overall candidate execution plan."""

    plan_id: str = Field(..., description="Unique plan identifier")
    steps: list[CandidateToolStep] = Field(default_factory=list, description="Ordered tool steps")
    reasoning: str = Field(default="", description="Reasoning for decomposing this request")

    @field_validator("steps")
    @classmethod
    def validate_steps_length(cls, v: list[CandidateToolStep]) -> list[CandidateToolStep]:
        if len(v) > 4:
            raise ValueError(f"Plan exceeds maximum limit of 4 steps (got {len(v)})")
        return v
