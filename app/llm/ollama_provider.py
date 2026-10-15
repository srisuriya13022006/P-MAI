import logging
from typing import Any
from ollama import Client

from app.core.config import settings
from app.llm.prompt_budget import budget_messages

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """Raised when an LLM prompt exceeds the available model context window."""

    def __init__(
        self,
        message: str,
        provider: str = "ollama",
        n_prompt_tokens: int | None = None,
        n_ctx: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx


def is_context_overflow_error(exc: Exception) -> bool:
    """Identify if an exception represents an Ollama or LLM context size limit exceeded."""
    if isinstance(exc, ContextOverflowError):
        return True
    err_str = str(exc).lower()
    if "exceed_context_size_error" in err_str:
        return True
    if "exceeds the available context size" in err_str:
        return True
    if "context size" in err_str and "exceed" in err_str:
        return True
    err_dict = getattr(exc, "error", None)
    if isinstance(err_dict, dict) and err_dict.get("type") == "exceed_context_size_error":
        return True
    return False


class OllamaProvider:
    """Handles communication with a local Ollama model."""

    provider_name: str = "ollama"
    provider_type: str = "local"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        num_ctx: int | None = None,
    ):
        self.model = model or settings.local_model
        self.client = Client(host=base_url or settings.ollama_base_url)
        self.num_ctx = num_ctx or getattr(settings, "ollama_num_ctx", 4096)
        self.context_limit = self.num_ctx
        self.reserved_response_tokens = getattr(settings, "local_llm_reserved_response_tokens", 512)
        self.last_overflow_recovered: bool = False

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        stage: str = "conversation",
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using the supplied conversation messages with prompt-budget management.
        Guarantees that context overflows are reduced before submission and provides controlled single retry.
        """
        self.last_overflow_recovered = False
        reserved = max_tokens or self.reserved_response_tokens
        max_prompt_budget = max(400, self.num_ctx - reserved)

        # 1. Proactive prompt budget enforcement
        budgeted_messages, budgeted_system, _ = budget_messages(
            messages=messages,
            system_prompt=system_prompt,
            max_prompt_tokens=max_prompt_budget,
            reserved_response_tokens=reserved,
            emergency_reduction=False,
        )

        chat_messages = []
        if budgeted_system:
            chat_messages.append({"role": "system", "content": budgeted_system})
        chat_messages.extend(budgeted_messages)

        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens

        try:
            response = self.client.chat(
                model=self.model,
                messages=chat_messages,
                options=options,
            )
            return response["message"]["content"]
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise exc

            logger.warning(
                "Ollama context overflow encountered (n_ctx=%d). Triggering controlled single reduction retry...",
                self.num_ctx,
            )

            # 2. Controlled overflow recovery: reduce request context once
            emergency_budget = max(300, int(self.num_ctx * 0.55))
            reduced_messages, reduced_system, _ = budget_messages(
                messages=messages,
                system_prompt=system_prompt,
                max_prompt_tokens=emergency_budget,
                reserved_response_tokens=reserved,
                emergency_reduction=True,
            )

            reduced_chat = []
            if reduced_system:
                reduced_chat.append({"role": "system", "content": reduced_system})
            reduced_chat.extend(reduced_messages)

            try:
                retry_response = self.client.chat(
                    model=self.model,
                    messages=reduced_chat,
                    options=options,
                )
                self.last_overflow_recovered = True
                logger.info("Ollama controlled context recovery retry succeeded.")
                return retry_response["message"]["content"]
            except Exception as retry_exc:
                if is_context_overflow_error(retry_exc):
                    logger.error(
                        "Ollama context overflow recovery failed after single retry. Returning clear recoverable error."
                    )
                    return (
                        "The conversation has exceeded the local model's available context memory. "
                        "Please ask a more focused question or start a fresh conversation."
                    )
                raise retry_exc


_canonical_ollama_provider: OllamaProvider | None = None


def get_ollama_provider() -> OllamaProvider:
    """Return the shared canonical OllamaProvider instance."""
    global _canonical_ollama_provider
    if _canonical_ollama_provider is None:
        _canonical_ollama_provider = OllamaProvider()
    return _canonical_ollama_provider