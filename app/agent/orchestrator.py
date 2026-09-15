from app.agent.policy import apply_policy
from app.agent.router import RequestAnalyzer
from app.llm.ollama_provider import OllamaProvider
from app.prompts.conversation import MAI_SYSTEM_PROMPT
from app.tools.calculator.tool import CalculatorTool
from app.tools.datetime.tool import DateTimeTool
from app.tools.registry import ToolRegistry
from app.agent.tool_argument_resolver import (
    resolve_calculator_arguments,
)


class MAIOrchestrator:
    """Controls the high-level MAI request workflow."""

    def __init__(self):
        self.router = RequestAnalyzer()
        self.llm = OllamaProvider()

        self.tool_registry = ToolRegistry()
        self.tool_registry.register(
            CalculatorTool()
        )
        self.tool_registry.register(
         DateTimeTool()
        )

    def handle(
        self,
        user_message: str,
        conversation_messages: list[dict[str, str]],
    ) -> str:

        decision = self.router.analyze(user_message)

        decision = apply_policy(
            user_message=user_message,
            decision=decision,
        )

        print("DECISION:", decision)
        print("TOOL ARGUMENTS:", decision.tool_arguments)


     

        if decision.route == "local":
            return self.llm.generate(
                conversation_messages,
                system_prompt=MAI_SYSTEM_PROMPT,
            )

        if decision.route == "tool":
            return self._execute_tool(
                decision,
                user_message,
            )

        if decision.route == "memory":
            return (
                "I understand that this request requires "
                "memory retrieval, but the memory subsystem "
                "is not connected yet."
            )

        if decision.route == "cloud":
            return (
                "I understand that this request requires a "
                "stronger reasoning model, but the cloud model "
                "is not connected yet."
            )

        if decision.route == "clarification":
            return (
                "I need a little more information before I "
                "can help with that."
            )

        return "I couldn't determine how to handle that request."

    def _execute_tool(
         self,
         decision,
         user_message: str,
    ) -> str:

        if not decision.tools:
            return "I couldn't determine which tool to use."

        tool_name = decision.tools[0]

        if not self.tool_registry.has(tool_name):
            return (
                f"The requested tool '{tool_name}' "
                "is not available."
            )

        tool = self.tool_registry.get(tool_name)

        print("TOOL:", tool.name)
        print("ARGS:", decision.tool_arguments)

        tool_arguments = decision.tool_arguments

        if tool_name == "calculator":
            tool_arguments = resolve_calculator_arguments(
                user_message
            )

        print("TOOL ARGUMENTS BEFORE RUN:", tool_arguments)

        result = tool.run(
            **tool_arguments
        )

        if not result.success:
            return (
                "I couldn't complete that operation."
            )

        return self._format_tool_result(result)

    def _format_tool_result(self, result) -> str:
        if result.tool_name == "calculator":
            value = result.data["result"]

            return (
                f"The calculation result is {value}."
            )

        if result.tool_name == "datetime":
            date = result.data["date"]
            time = result.data["time"]
            day = result.data["day"]
            timezone = result.data["timezone"]

            return (
                f"Today is {day}, {date}, "
                f"and the current time is {time} ({timezone})."
            )

        return str(result.data)