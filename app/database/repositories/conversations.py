from sqlalchemy.orm import Session

from app.database.models import Conversation


class ConversationRepository:
    """Database operations related to conversations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(
        self,
        conversation_id: str,
    ) -> Conversation | None:
        return self.db.get(
            Conversation,
            conversation_id,
        )

    def create(
        self,
        conversation_id: str,
        user_id: str,
        title: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title,
        )

        self.db.add(conversation)
        self.db.flush()

        return conversation

    def update_summary(self, conversation: Conversation, summary: str) -> None:
        conversation.summary = summary