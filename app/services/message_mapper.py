from app.database.models import Message


def to_llm_message(message: Message) -> dict[str, str]:
    """Convert a database message into the format expected by the LLM."""

    return {
        "role": message.role,
        "content": message.content,
    }


def to_llm_messages(
    messages: list[Message],
) -> list[dict[str, str]]:
    """Convert database messages into LLM chat messages."""

    return [
        to_llm_message(message)
        for message in messages
    ]