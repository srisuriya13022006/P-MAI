from app.llm.ollama_provider import OllamaProvider
from app.prompts.router import ROUTER_SYSTEM_PROMPT
from app.schemas.agent import AgentDecision


class RequestAnalyzer:
    """Analyzes user requests and produces a structured MAI decision."""

    def __init__(self):
        self.llm = OllamaProvider()

    def analyze(self, user_message: str) -> AgentDecision:
        """Analyze a user request and return a validated decision."""

        response = self.llm.client.chat(
            model=self.llm.model,
            messages=[
                {
                    "role": "system",
                    "content": ROUTER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            format=AgentDecision.model_json_schema(),
        )

        content = response["message"]["content"]

        return AgentDecision.model_validate_json(content)