from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.memory.service import MemoryService


def test_memory_is_filtered_retrieved_and_updated():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        service = MemoryService(MemoryRepository(db))

        assert service.remember_from_message("user-1", "I prefer C++.") == 1
        assert service.remember_from_message("user-1", "I ate dosa for lunch.") == 0
        db.commit()

        context = service.relevant_context(
            "user-1",
            "What programming language do I prefer?",
        )
        assert context == ["User prefers C++"]

        assert service.remember_from_message("user-1", "I prefer Python.") == 1
        db.commit()
        memories = MemoryRepository(db).list_for_user("user-1")
        assert [memory.content for memory in memories] == ["User prefers Python"]

        assert service.forget("user-1", memories[0].id) is True
        db.commit()
        assert service.relevant_context("user-1", "What language do I prefer?") == []