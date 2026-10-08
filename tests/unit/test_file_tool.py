from pathlib import Path

from app.agent.policy import apply_policy
from app.schemas.agent import AgentDecision
from app.tools.filesystem.tool import FileSearchTool


def test_policy_routes_document_questions_to_file_search():
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )

    routed = apply_policy(
        "Find the note about onboarding in the project files.",
        decision.model_copy(),
    )

    assert routed.route == "tool"
    assert routed.tools == ["file_search"]
    assert "onboarding" in routed.tool_arguments["query"].lower()


def test_file_search_tool_reads_project_documents(tmp_path):
    note = tmp_path / "onboarding.md"
    note.write_text("# Onboarding\n\nWelcome to the team. Follow the setup guide.", encoding="utf-8")

    tool = FileSearchTool(root_dir=str(tmp_path))
    result = tool.run(query="onboarding setup guide")

    assert result.success is True
    assert len(result.data["matches"]) >= 1
    assert "Onboarding" in result.data["matches"][0]["title"]
