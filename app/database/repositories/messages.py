from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Message


class MessageRepository:
    """Database operations related to messages."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        self.db.add(message)
        self.db.flush()

        return message

    def get_for_conversation(
        self,
        conversation_id: str,
    ) -> list[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )

        return list(self.db.scalars(statement).all())