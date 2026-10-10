from ollama import Client

from app.core.config import settings


class OllamaProvider:
    """Handles communication with a local Ollama model."""

    provider_name: str = "ollama"
    provider_type: str = "local"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.local_model
        self.client = Client(host=base_url or settings.ollama_base_url)

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


_canonical_ollama_provider: OllamaProvider | None = None


def get_ollama_provider() -> OllamaProvider:
    """Return the shared canonical OllamaProvider instance."""
    global _canonical_ollama_provider
    if _canonical_ollama_provider is None:
        _canonical_ollama_provider = OllamaProvider()
    return _canonical_ollama_provider