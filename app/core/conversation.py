from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Represents the message history of one MAI conversation."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages.copy() 