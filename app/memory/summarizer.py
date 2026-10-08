from app.llm.ollama_provider import OllamaProvider


class ConversationSummarizer:
    """Creates a compact, factual summary for older conversation turns."""

    def __init__(self, llm: OllamaProvider | None = None) -> None:
        self.llm = llm or OllamaProvider()

    def summarize(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )
        return self.llm.generate(
            [{"role": "user", "content": transcript}],
            system_prompt=(
                "Summarize this conversation for future context. "
                "Keep only stable facts, decisions, goals, open tasks, "
                "and important preferences. Be concise and factual."
            ),
        )