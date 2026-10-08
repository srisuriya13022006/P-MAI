from typing import Literal

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    content: str = Field(min_length=1)
    memory_type: Literal["fact", "preference", "goal", "project"]
    importance: int = Field(default=1, ge=1, le=3)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    memory_type: Literal["fact", "preference", "goal", "project"] = "fact"
    importance: int = Field(default=2, ge=1, le=3)


class MemoryUpdateRequest(BaseModel):
    content: str = Field(min_length=1)


class MemoryResponse(BaseModel):
    id: int
    content: str
    memory_type: str
    importance: int
    confidence: float