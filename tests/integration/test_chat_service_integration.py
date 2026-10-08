from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.connection import Base
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.messages import MessageRepository
from app.memory.service import MemoryService
from app.services.chat_service import ChatService


def test_chat_service_persists_conversation_and_response():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        service = ChatService(db)
        service.orchestrator.router = type(
            "Router",
            (),
            {"analyze": lambda self, message: type("Decision", (), {"route": "local", "tools": [], "tool_arguments": {}, "needs_clarification": False})()},
        )()
        service.orchestrator.llm = type(
            "FakeLLM",
            (),
            {"generate": lambda self, messages, system_prompt=None: "hello there"},
        )()

        response = service.chat(
            conversation_id="chat-1",
            user_id="user-1",
            user_message="Hi there!",
        )

        assert response == "hello there"

        conversation = ConversationRepository(db).get_by_id("chat-1")
        messages = MessageRepository(db).get_for_conversation(conversation.id)

        assert conversation is not None
        assert len(messages) == 2
        assert messages[0].content == "Hi there!"
        assert messages[1].content == "hello there"


def test_chat_service_builds_memory_context_from_user_memory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        service = ChatService(db)
        service.memory_service.remember("user-1", "User likes coffee")
        db.commit()

        service.orchestrator.router = type(
            "Router",
            (),
            {"analyze": lambda self, message: type("Decision", (), {"route": "local", "tools": [], "tool_arguments": {}, "needs_clarification": False})()},
        )()
        service.orchestrator.llm = type(
            "FakeLLM",
            (),
            {"generate": lambda self, messages, system_prompt=None: "I remember you like coffee."},
        )()

        context = service.memory_service.relevant_context("user-1", "What do I like?")
        assert "User likes coffee" in context

        response = service.chat(
            conversation_id="chat-2",
            user_id="user-1",
            user_message="What do I like?",
        )

        assert "coffee" in response.lower()
