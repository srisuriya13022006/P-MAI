from typing import Any, Literal

from pydantic import BaseModel, Field

from app.memory.service import MemoryService
from app.tools.base import BaseTool
from app.tools.result import ToolResult


class RememberMemoryInput(BaseModel):
    content: str = Field(min_length=1)
    memory_type: Literal["fact", "preference", "goal", "project"] = "fact"
    importance: int = Field(default=2, ge=1, le=3)


class SearchMemoryInput(BaseModel):
    query: str = Field(min_length=1)


class UpdateMemoryInput(BaseModel):
    memory_id: int = Field(gt=0)
    content: str = Field(min_length=1)


class ForgetMemoryInput(BaseModel):
    memory_id: int | None = Field(default=None, gt=0)
    query: str | None = Field(default=None, min_length=1)


class _MemoryTool(BaseTool):
    def __init__(self, memory_service: MemoryService, user_id: str) -> None:
        self.memory_service = memory_service
        self.user_id = user_id

    def set_user_id(self, user_id: str) -> None:
        self.user_id = user_id


class RememberMemoryTool(_MemoryTool):
    @property
    def name(self) -> str:
        return "remember_memory"

    @property
    def description(self) -> str:
        return "Save an explicit user fact, preference, goal, or project."

    @property
    def input_schema(self) -> type[BaseModel]:
        return RememberMemoryInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            memory = self.memory_service.remember(
                user_id=self.user_id,
                content=arguments.content,
                memory_type=arguments.memory_type,
                importance=arguments.importance,
            )
            return ToolResult(self.name, True, {"memory_id": memory.id})
        except Exception as exc:
            return ToolResult(self.name, False, error=str(exc))


class SearchMemoryTool(_MemoryTool):
    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return "Search the user's stored long-term memories."

    @property
    def input_schema(self) -> type[BaseModel]:
        return SearchMemoryInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            memories = self.memory_service.search(self.user_id, arguments.query)
            return ToolResult(
                self.name,
                True,
                {"memories": [{"id": item.id, "content": item.content} for item in memories]},
            )
        except Exception as exc:
            return ToolResult(self.name, False, error=str(exc))


class UpdateMemoryTool(_MemoryTool):
    @property
    def name(self) -> str:
        return "update_memory"

    @property
    def description(self) -> str:
        return "Update one of the user's stored memories."

    @property
    def input_schema(self) -> type[BaseModel]:
        return UpdateMemoryInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            updated = self.memory_service.update(
                self.user_id,
                arguments.memory_id,
                arguments.content,
            )
            return ToolResult(self.name, updated is not None, {"memory_id": arguments.memory_id})
        except Exception as exc:
            return ToolResult(self.name, False, error=str(exc))


class ForgetMemoryTool(_MemoryTool):
    @property
    def name(self) -> str:
        return "forget_memory"

    @property
    def description(self) -> str:
        return "Forget one of the user's stored memories by id or search text."

    @property
    def input_schema(self) -> type[BaseModel]:
        return ForgetMemoryInput

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            arguments = self.input_schema.model_validate(kwargs)
            deleted = self.memory_service.forget(
                self.user_id,
                memory_id=arguments.memory_id,
                query=arguments.query,
            )
            return ToolResult(self.name, deleted, {"deleted": deleted})
        except Exception as exc:
            return ToolResult(self.name, False, error=str(exc))