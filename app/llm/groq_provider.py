import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on optional cloud dependency
    OpenAI = None

from app.core.config import settings


class GroqProvider:
    """Cloud model provider using Groq's OpenAI-compatible API."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for Groq cloud inference. "
                "Install it with: pip install openai"
            )

        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the project environment.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)

        response = self.client.responses.create(
            model=self.model,
            input=request_messages,
        )
        return getattr(response, "output_text", "")
