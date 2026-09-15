from sqlalchemy import delete, select

from app.database.connection import SessionLocal
from app.database.models import Conversation, Message
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.messages import MessageRepository


def main() -> None:
    db = SessionLocal()

    conversation_id = "repo-test-1"
    user_id = "user-1"

    try:
        # Clean previous test data
        db.execute(
            delete(Message).where(
                Message.conversation_id == conversation_id
            )
        )

        db.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id
            )
        )

        db.commit()

        conversation_repo = ConversationRepository(db)
        message_repo = MessageRepository(db)

        # Create conversation
        conversation = conversation_repo.create(
            conversation_id=conversation_id,
            user_id=user_id,
            title="Repository Test",
        )

        # Create messages
        message_repo.create(
            conversation_id=conversation.id,
            role="user",
            content="My name is Suriya.",
        )

        message_repo.create(
            conversation_id=conversation.id,
            role="assistant",
            content="Nice to meet you, Suriya!",
        )

        db.commit()

        # Read back
        loaded_conversation = conversation_repo.get_by_id(
            conversation_id
        )

        messages = message_repo.get_for_conversation(
            conversation_id
        )

        print("Conversation:")
        print(loaded_conversation.id)

        print("\nMessages:")

        for message in messages:
            print(f"{message.role}: {message.content}")

        print("\nRepository test successful.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()