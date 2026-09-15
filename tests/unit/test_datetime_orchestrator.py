from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.orchestrator import MAIOrchestrator
from app.schemas.agent import AgentDecision
from app.tools.datetime.tool import DateTimeTool


def test_orchestrator_answers_location_datetime_request():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="datetime",
                route="tool",
                needs_clarification=False,
                tools=["datetime"],
                tool_arguments={"location": "New York"},
                reason="datetime request",
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

    response = orchestrator.handle(
        "What time is it in New York?",
        [],
    )

    assert "08:00:00" in response
    assert "America/New_York" in response