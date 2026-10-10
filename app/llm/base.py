"""
P15.4 — LLM Provider Interface.
Defines unified protocol for primary cloud and fallback local LLM providers.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol satisfied by all MAI LLM providers."""

    @property
    def provider_name(self) -> str:
        """Identifier name of the LLM provider (e.g. 'groq', 'ollama', 'mock')."""
        ...

    @property
    def provider_type(self) -> str:
        """Provider tier: 'cloud' or 'local'."""
        ...

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate conversational/reasoning completion.
        """
        ...
