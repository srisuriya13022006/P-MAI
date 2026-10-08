from app.agent.orchestrator import MAIOrchestrator
from app.agent.permissions import PermissionManager


def test_permission_manager_requires_confirmation_for_write_actions():
    manager = PermissionManager()

    decision = manager.check(
        tool_name="remember_memory",
        user_id="user-1",
        confirmed=False,
    )

    assert decision.requires_confirmation is True
    assert decision.allowed is False


def test_orchestrator_blocks_write_tool_until_confirmed():
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {"analyze": lambda self, message: type("Decision", (), {"route": "tool", "tools": ["remember_memory"], "tool_arguments": {"content": "I like Python."}, "needs_clarification": False})()},
    )()

    response = orchestrator.handle(
        "Remember that I like Python.",
        [],
        user_id="user-1",
    )

    assert "confirmation" in response.lower()
    assert "remember" in response.lower()
