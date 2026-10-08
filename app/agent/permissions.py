from dataclasses import dataclass
from typing import Any


WRITE_TOOLS = {
    "remember_memory",
    "update_memory",
    "forget_memory",
}


@dataclass
class PermissionDecision:
    tool_name: str
    allowed: bool
    requires_confirmation: bool
    reason: str
    confirmed: bool = False


class PermissionManager:
    """Guards mutating tools behind explicit confirmation."""

    def check(
        self,
        tool_name: str,
        user_id: str | None = None,
        confirmed: bool = False,
        **_: Any,
    ) -> PermissionDecision:
        if tool_name not in WRITE_TOOLS:
            return PermissionDecision(
                tool_name=tool_name,
                allowed=True,
                requires_confirmation=False,
                reason="Read-only tool does not require confirmation.",
                confirmed=confirmed,
            )

        if confirmed:
            return PermissionDecision(
                tool_name=tool_name,
                allowed=True,
                requires_confirmation=False,
                reason="User explicitly confirmed this write action.",
                confirmed=True,
            )

        return PermissionDecision(
            tool_name=tool_name,
            allowed=False,
            requires_confirmation=True,
            reason="This action changes stored data and needs explicit user confirmation.",
            confirmed=False,
        )
