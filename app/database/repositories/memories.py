from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.database.models import Memory


class MemoryRepository:
    """Database operations for durable user memories."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, user_id: str, **values) -> Memory:
        memory = Memory(user_id=user_id, **values)
        self.db.add(memory)
        self.db.flush()
        return memory

    def get(self, user_id: str, memory_id: int) -> Memory | None:
        statement = select(Memory).where(
            Memory.user_id == user_id,
            Memory.id == memory_id,
        )
        return self.db.scalar(statement)

    def list_for_user(self, user_id: str, include_inactive: bool = False) -> list[Memory]:
        conditions = [Memory.user_id == user_id]
        if not include_inactive:
            conditions.append(Memory.is_active.is_(True))
        statement = (
            select(Memory)
            .where(*conditions)
            .order_by(Memory.importance.desc(), Memory.updated_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def find_by_content(self, user_id: str, content: str, include_inactive: bool = False) -> Memory | None:
        conditions = [Memory.user_id == user_id, Memory.content == content]
        if not include_inactive:
            conditions.append(Memory.is_active.is_(True))
        statement = select(Memory).where(*conditions)
        return self.db.scalar(statement)

    def find_by_type(self, user_id: str, memory_type: str, include_inactive: bool = False) -> Memory | None:
        conditions = [Memory.user_id == user_id, Memory.memory_type == memory_type]
        if not include_inactive:
            conditions.append(Memory.is_active.is_(True))
        statement = (
            select(Memory)
            .where(*conditions)
            .order_by(Memory.updated_at.desc())
        )
        return self.db.scalars(statement).first()

    def mark_superseded(self, memory: Memory, superseded_by_id: int) -> None:
        memory.is_active = False
        memory.superseded_by = superseded_by_id
        memory.updated_at = datetime.utcnow()

    def touch_accessed(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        statement = (
            update(Memory)
            .where(Memory.id.in_(memory_ids))
            .values(last_accessed_at=datetime.utcnow())
        )
        self.db.execute(statement)

    def delete(self, user_id: str, memory_id: int) -> bool:
        statement = delete(Memory).where(
            Memory.user_id == user_id,
            Memory.id == memory_id,
        )
        result = self.db.execute(statement)
        return result.rowcount > 0

    def find_matching(self, user_id: str, query: str) -> Memory | None:
        statement = (
            select(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.is_active.is_(True),
                Memory.content.ilike(f"%{query}%"),
            )
            .limit(1)
        )
        return self.db.scalar(statement)