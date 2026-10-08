from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.repositories.memories import MemoryRepository
from app.database.session import get_db
from app.memory.service import MemoryService
from app.schemas.memory import (
    MemoryCreateRequest,
    MemoryResponse,
    MemoryUpdateRequest,
)


router = APIRouter(prefix="/memory", tags=["memory"])


def _service(db: Session) -> MemoryService:
    return MemoryService(MemoryRepository(db))


def _response(memory) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        memory_type=memory.memory_type,
        importance=memory.importance,
        confidence=memory.confidence,
    )


@router.get("/{user_id}", response_model=list[MemoryResponse])
def list_memories(user_id: str, db: Session = Depends(get_db)):
    return [_response(item) for item in _service(db).repository.list_for_user(user_id)]


@router.post("/{user_id}", response_model=MemoryResponse)
def create_memory(
    user_id: str,
    request: MemoryCreateRequest,
    db: Session = Depends(get_db),
):
    memory = _service(db).remember(
        user_id=user_id,
        content=request.content,
        memory_type=request.memory_type,
        importance=request.importance,
    )
    db.commit()
    return _response(memory)


@router.patch("/{user_id}/{memory_id}", response_model=MemoryResponse)
def update_memory(
    user_id: str,
    memory_id: int,
    request: MemoryUpdateRequest,
    db: Session = Depends(get_db),
):
    memory = _service(db).update(user_id, memory_id, request.content)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    db.commit()
    return _response(memory)


@router.delete("/{user_id}/{memory_id}", status_code=204)
def delete_memory(user_id: str, memory_id: int, db: Session = Depends(get_db)):
    if not _service(db).forget(user_id, memory_id=memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    db.commit()