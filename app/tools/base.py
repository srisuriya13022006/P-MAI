from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from app.tools.result import ToolResult


class BaseTool(ABC):
    """Base contract for all MAI tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Description shown to the agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """Pydantic schema used to validate tool arguments."""
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool."""
        raise NotImplementedError