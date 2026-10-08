from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.orchestrator import MAIOrchestrator
from app.agent.policy import apply_policy
from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.memory.service import MemoryService
from app.schemas.agent import AgentDecision
from app.tools.memory.tool import SearchMemoryTool


def test_memory_tools_are_registered_and_explicit_requests_are_routed():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        service = MemoryService(MemoryRepository(db))
        orchestrator = MAIOrchestrator(service)

        assert orchestrator.tool_registry.has("remember_memory")
        assert orchestrator.tool_registry.has("search_memory")
        assert orchestrator.tool_registry.has("update_memory")
        assert orchestrator.tool_registry.has("forget_memory")

    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )

    remember = apply_policy("Remember that I prefer C++.", decision.model_copy())
    assert remember.tools == ["remember_memory"]
    assert remember.tool_arguments["content"] == "I prefer C++"

    search = apply_policy("Search my memories for C++.", decision.model_copy())
    assert search.tools == ["search_memory"]

    forget = apply_policy("Forget that I prefer C++.", decision.model_copy())
    assert forget.tools == ["forget_memory"]

    update = apply_policy("Update memory 7 to User prefers Python.", decision.model_copy())
    assert update.tools == ["update_memory"]
    assert update.tool_arguments == {
        "memory_id": 7,
        "content": "User prefers Python",
    }

    with Session(engine) as db:
        service = MemoryService(MemoryRepository(db))
        service.remember(
            "user-1",
            "User prefers C++",
            memory_type="preference",
        )
        db.commit()
        result = SearchMemoryTool(service, "user-1").run(
            query="What language do I prefer?",
        )
        assert result.success is True