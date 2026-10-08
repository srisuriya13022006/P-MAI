from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.orchestrator import MAIOrchestrator
from app.schemas.agent import AgentDecision
from app.tools.datetime.tool import DateTimeTool


def test_orchestrator_executes_multiple_tools_in_sequence():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="tool",
                route="tool",
                needs_clarification=False,
                tools=["calculator", "datetime"],
                tool_arguments={
                    "expression": "2 + 3",
                    "location": "New York",
                },
                reason="multi-tool request",
            )
        },
    )()
    orchestrator.tool_registry._tools["datetime"] = DateTimeTool(
        clock=lambda timezone: datetime(
            2026,
            9,
            15,
            12,
            tzinfo=ZoneInfo("UTC"),
        ).astimezone(timezone),
    )
    orchestrator.llm = type(
        "FakeLLM",
        (),
        {
            "generate": lambda self, messages, system_prompt=None: (
                "The final answer is 5 and the time in New York is 08:00:00."
            )
        },
    )()

    response = orchestrator.handle(
        "Calculate 2 + 3 and tell me the time in New York.",
        [],
    )

    assert "5" in response
    assert "08:00:00" in response
    assert "New York" in response


def test_orchestrator_handles_single_tool_requests():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="arithmetic",
                route="tool",
                needs_clarification=False,
                tools=["calculator"],
                tool_arguments={"expression": "7 * 8"},
                reason="arithmetic request",
            )
        },
    )()

    response = orchestrator.handle("What is 7 * 8?", [])

    assert "56" in response


def test_orchestrator_uses_local_model_for_simple_tool_queries():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="arithmetic",
                route="tool",
                needs_clarification=False,
                tools=["calculator"],
                tool_arguments={"expression": "7 * 8"},
                reason="simple arithmetic request",
            )
        },
    )()

    calls = []

    class FakeLocalLLM:
        def generate(self, messages, system_prompt=None):
            calls.append("local")
            return "The calculation result is 56."

    class FakeCloudLLM:
        def generate(self, messages, system_prompt=None):
            calls.append("cloud")
            return "cloud fallback"

    orchestrator.llm = FakeLocalLLM()
    orchestrator.cloud_llm = FakeCloudLLM()

    response = orchestrator.handle("What is 7 * 8?", [])

    assert "56" in response
    assert calls == []


def test_orchestrator_uses_cloud_model_for_complex_web_search_queries():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="web_search",
                route="tool",
                needs_clarification=False,
                tools=["web_search"],
                tool_arguments={"query": "Search the web for public profile and latest updates about Srisuriya at Sri Eshwar College"},
                reason="complex public research request",
            )
        },
    )()

    calls = []

    class FakeWebTool:
        name = "web_search"

        def run(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": {
                        "query": kwargs["query"],
                        "results": [
                            {"title": "Public profile info", "url": "https://example.com/profile", "snippet": "Verified public card"},
                            {"title": "College page", "url": "https://example.com/college", "snippet": "Academic profile"},
                        ],
                    },
                    "error": None,
                    "tool_name": "web_search",
                },
            )()

    class FakeLocalLLM:
        def generate(self, messages, system_prompt=None):
            calls.append("local")
            return "local fallback answer"

    class FakeCloudLLM:
        def generate(self, messages, system_prompt=None):
            calls.append("cloud")
            return "cloud research synthesis based on the web results"

    orchestrator.tool_registry._tools["web_search"] = FakeWebTool()
    orchestrator.llm = FakeLocalLLM()
    orchestrator.cloud_llm = FakeCloudLLM()

    response = orchestrator.handle(
        "Search the web for public profile and latest updates about Srisuriya at Sri Eshwar College",
        [],
    )

    assert "cloud research synthesis" in response
    assert calls == ["cloud"]