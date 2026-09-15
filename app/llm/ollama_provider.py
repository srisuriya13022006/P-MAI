from ollama import Client

from app.core.config import settings


class OllamaProvider:
    """Handles communication with a local Ollama model."""

    def __init__(self):
        self.model = settings.local_model
        self.client = Client(host=settings.ollama_base_url)

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using the supplied conversation messages."""

        chat_messages = []

        if system_prompt:
            chat_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        chat_messages.extend(messages)

        response = self.client.chat(
            model=self.model,
            messages=chat_messages,
        )

        return response["message"]["content"]