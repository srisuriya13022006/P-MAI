from sqlalchemy.orm import Session

from app.agent.orchestrator import MAIOrchestrator
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.messages import MessageRepository
from app.services.message_mapper import to_llm_messages


class ChatService:
    """Coordinates persistent MAI conversations."""

    def __init__(self, db: Session):
        self.db = db

        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)

        self.orchestrator = MAIOrchestrator()

    def chat(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
    ) -> str:

        conversation = self.conversation_repo.get_by_id(
            conversation_id
        )

        if conversation is None:
            conversation = self.conversation_repo.create(
                conversation_id=conversation_id,
                user_id=user_id,
                title=None,
            )

        # Store the new user message.
        self.message_repo.create(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )

        # Load complete conversation history.
        messages = self.message_repo.get_for_conversation(
            conversation.id
        )

        llm_messages = to_llm_messages(messages)

        # Let MAI decide how to handle the request.
        response = self.orchestrator.handle(
            user_message=user_message,
            conversation_messages=llm_messages,
        )

        # Store the assistant response.
        self.message_repo.create(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
        )

        self.db.commit()

        return response