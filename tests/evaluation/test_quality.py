from app.agent.orchestrator import MAIOrchestrator


class FakeRouter:
    def analyze(self, message: str):
        text = message.lower()
        if "remember" in text:
            return type(
                "Decision",
                (),
                {
                    "route": "tool",
                    "tools": ["remember_memory"],
                    "tool_arguments": {"content": "I like Python."},
                    "needs_clarification": False,
                },
            )()
        if "time" in text or "date" in text:
            return type(
                "Decision",
                (),
                {
                    "route": "tool",
                    "tools": ["datetime"],
                    "tool_arguments": {"location": "New York"},
                    "needs_clarification": False,
                },
            )()
        return type(
            "Decision",
            (),
            {
                "route": "local",
                "tools": [],
                "tool_arguments": {},
                "needs_clarification": False,
            },
        )()


def test_quality_suite_smoke_checks_core_paths():
    orchestrator = MAIOrchestrator()
    orchestrator.router = FakeRouter()

    remember_result = orchestrator.handle(
        "Remember that I like Python.",
        [],
        user_id="user-1",
    )
    assert "confirmation" in remember_result.lower()

    time_result = orchestrator.handle(
        "What time is it in New York?",
        [],
    )
    assert "Today is" in time_result or "08:00:00" in time_result or "America/New_York" in time_result
