from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """Structured decision produced by MAI's request analyzer."""

    intent: str = Field(
        description="The user's primary intent."
    )

    route: Literal[
        "local",
        "tool",
        "memory",
        "cloud",
        "clarification",
    ]

    needs_clarification: bool

    tools: list[str] = Field(
        default_factory=list,
        description="Tools required to fulfill the request.",
    )

    tool_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments required by the selected tools.",
    )

    reason: str