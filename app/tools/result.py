from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Standard result returned by every MAI tool."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None