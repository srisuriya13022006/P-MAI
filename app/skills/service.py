"""
P16 — Procedural Skill Service.
Manages the safe lifecycle, proposal generation, retrieval, and execution of procedural skills.
"""
import logging
from typing import Any

from app.agent.task_state import ActiveTaskState
from app.database.models import Skill
from app.database.repositories.skills import SkillRepository
from app.skills.executor import SkillExecutor
from app.skills.model import (
    AdaptedPlanModel,
    SkillExecutionResult,
    SkillProposal,
    SkillSecurityViolationError,
    SkillValidationError,
    WorkflowSummaryModel,
    validate_skill_security,
)
from app.skills.nemotron_assistant import NemotronSkillAssistant
from app.skills.retriever import SkillRetrievalOutcome, SkillRetriever
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillService:
    """Coordinates procedural memory operations with strict user scoping and safety limits."""

    def __init__(
        self,
        repository: SkillRepository,
        tool_registry: ToolRegistry,
        retriever: SkillRetriever | None = None,
        procedural_llm: Any | None = None,
        nemotron_assistant: NemotronSkillAssistant | None = None,
    ) -> None:
        self.repository = repository
        self.tool_registry = tool_registry
        self.retriever = retriever or SkillRetriever()
        self.executor = SkillExecutor(tool_registry=tool_registry, skill_repo=repository)
        self.procedural_llm = procedural_llm
        if nemotron_assistant is not None:
            self.nemotron_assistant = nemotron_assistant
        elif procedural_llm is not None:
            self.nemotron_assistant = NemotronSkillAssistant(llm=procedural_llm, tool_registry=tool_registry)
        else:
            self.nemotron_assistant = None

    def propose_from_task(
        self,
        active_task: ActiveTaskState,
        name: str,
        description: str | None = None,
        intended_use: str | None = None,
        use_llm: bool = True,
    ) -> SkillProposal:
        """
        Construct a SkillProposal from a completed or active task without persisting it.
        Uses Nemotron assistant if available with deterministic fallback.
        Requires explicit user approval before saving.
        """
        if not active_task.completed_steps and not (active_task.current_plan and active_task.current_plan.steps):
            raise SkillValidationError("Cannot create a skill from a task with no executed tool steps.")

        # Attempt Nemotron LLM proposal if configured
        if use_llm and self.nemotron_assistant is not None:
            trace_dict = {
                "task_id": active_task.task_id,
                "user_goal": active_task.original_user_goal or name,
                "completed_steps": [
                    {
                        "step_id": getattr(s, "step_id", None) or (s.get("step_id") if isinstance(s, dict) else f"step_{idx}"),
                        "tool_name": getattr(s, "tool_name", "") if hasattr(s, "tool_name") else (s.get("tool_name", "") if isinstance(s, dict) else ""),
                        "arguments": getattr(s, "arguments", {}) if hasattr(s, "arguments") else (s.get("arguments", {}) if isinstance(s, dict) else {}),
                        "output_text": getattr(s, "output_text", "") if hasattr(s, "output_text") else (s.get("output_text", "") if isinstance(s, dict) else ""),
                    }
                    for idx, s in enumerate(active_task.completed_steps or [], 1)
                ],
            }
            try:
                proposal = self.nemotron_assistant.propose_skill_from_trace(
                    trace_data=trace_dict,
                    user_goal=active_task.original_user_goal,
                )
                return proposal
            except Exception as e:
                logger.warning("Nemotron skill proposal failed (%s); falling back to deterministic extraction.", e)


        # Determine steps from completed steps or current plan
        raw_steps: list[dict[str, Any]] = []
        schema: dict[str, Any] = {}

        if active_task.completed_steps:
            for idx, step_res in enumerate(active_task.completed_steps, 1):
                t_name = getattr(step_res, "tool_name", None) or (step_res.get("tool_name") if isinstance(step_res, dict) else "")
                t_args = getattr(step_res, "arguments", None) or (step_res.get("arguments") if isinstance(step_res, dict) else {})
                raw_steps.append({
                    "step_id": f"step_{idx}",
                    "tool_name": t_name,
                    "arguments": dict(t_args),
                    "depends_on": [f"step_{idx-1}"] if idx > 1 else [],
                    "purpose": f"Execute {t_name}",
                })
        elif active_task.current_plan:
            for s in active_task.current_plan.steps:
                raw_steps.append({
                    "step_id": s.step_id,
                    "tool_name": s.tool_name,
                    "arguments": dict(s.arguments),
                    "depends_on": list(s.depends_on),
                    "purpose": s.purpose,
                    "extractor": s.extractor,
                })

        # Detect parameterized inputs from goal or arguments
        for s in raw_steps:
            for k, v in s.get("arguments", {}).items():
                if isinstance(v, str) and not v.startswith("{"):
                    # Check if value appears in original goal
                    if active_task.original_user_goal and v.lower() in active_task.original_user_goal.lower():
                        param_name = k
                        schema[param_name] = {
                            "type": "string",
                            "required": True,
                            "description": f"Target for {k}",
                        }
                        s["arguments"][k] = f"{{{param_name}}}"

        clean_name = name.strip()
        clean_desc = (description or active_task.original_user_goal or clean_name).strip()
        clean_use = (intended_use or f"Reusable workflow for: {clean_desc}").strip()

        # Validate security before proposing
        validate_skill_security(clean_name, clean_desc, raw_steps, schema)

        return SkillProposal(
            name=clean_name,
            description=clean_desc,
            intended_use=clean_use,
            parameters_schema=schema,
            steps=raw_steps,
            completion_criteria=f"Successfully complete {len(raw_steps)} tool steps",
            provenance="derived_from_task",
            derived_from_task_id=active_task.task_id,
        )

    def validate_skill_definition(
        self,
        user_id: str,
        name: str,
        description: str,
        steps: list[dict[str, Any]],
        parameters_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a skill definition against tool registry, security, and step limits."""
        errors: list[str] = []
        try:
            validate_skill_security(name, description, steps, parameters_schema or {})
        except Exception as e:
            errors.append(str(e))

        if len(steps) > 10:
            errors.append(f"Step count {len(steps)} exceeds maximum allowed steps 10.")

        for s in steps:
            t_name = s.get("tool_name", "")
            is_mem = t_name in {"remember_memory", "search_memory", "update_memory", "forget_memory"}
            if not (self.tool_registry.has(t_name) or is_mem):
                errors.append(f"Tool '{t_name}' is not registered.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    def create_skill(
        self,
        user_id: str,
        name: str,
        description: str,
        steps: list[dict[str, Any]],
        parameters_schema: dict[str, Any] | None = None,
        intended_use: str | None = None,
        completion_criteria: str | None = None,
        provenance: str = "derived_from_task",
    ) -> Skill:
        """Validate security and persist a new user-approved skill."""
        if len(steps) > 10:
            raise SkillValidationError(f"Skill cannot exceed 10 steps (got {len(steps)}).")

        validate_skill_security(name, description, steps, parameters_schema)

        # Ensure tools exist
        for s in steps:
            t_name = s.get("tool_name", "")
            is_mem = t_name in {"remember_memory", "search_memory", "update_memory", "forget_memory"}
            if not (self.tool_registry.has(t_name) or is_mem):
                raise SkillValidationError(f"Cannot save skill with unregistered tool: '{t_name}'")

        return self.repository.create(
            user_id=user_id,
            name=name,
            description=description,
            steps=steps,
            parameters_schema=parameters_schema or {},
            intended_use=intended_use,
            completion_criteria=completion_criteria,
            provenance=provenance,
        )

    def get_skill(self, user_id: str, skill_id_or_name: str) -> Skill | None:
        """Find a skill by ID or case-insensitive name for an authorized user."""
        skill = self.repository.get(user_id, skill_id_or_name)
        if not skill:
            skill = self.repository.get_by_name(user_id, skill_id_or_name, active_only=False)
        return skill

    def list_skills(self, user_id: str, active_only: bool = True) -> list[Skill]:
        """List all skills for the user."""
        return self.repository.list_for_user(user_id=user_id, active_only=active_only)

    def search_skills(self, user_id: str, query: str) -> SkillRetrievalOutcome:
        """Search and resolve skills by query with ambiguity detection."""
        user_skills = self.repository.list_for_user(user_id=user_id, active_only=True)
        return self.retriever.resolve_skill(query=query, skills=user_skills)

    def execute_skill(
        self,
        user_id: str,
        skill_id_or_name: str,
        inputs: dict[str, Any] | None = None,
        active_task: ActiveTaskState | None = None,
    ) -> SkillExecutionResult:
        """Retrieve and execute a skill for the user."""
        inputs = inputs or {}
        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            raise ValueError(f"Skill not found: {skill_id_or_name}")

        if not skill.is_active:
            return SkillExecutionResult(
                skill_id=skill.id,
                skill_name=skill.name,
                success=False,
                steps_executed=0,
                duration_ms=0.0,
                output_summary=f"Skill '{skill.name}' is currently disabled.",
                error="Skill disabled",
            )

        # Validate inputs before execution
        self.executor.validate_and_filter_inputs(skill, inputs)

        return self.executor.execute(
            skill=skill,
            user_inputs=inputs,
            active_task=active_task,
            user_id=user_id,
        )

    def disable_skill(self, user_id: str, skill_id_or_name: str) -> bool:
        """Disable a skill so it cannot be executed until reactivated."""
        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            return False
        skill.is_active = False
        self.repository.update(skill)
        return True

    def reactivate_skill(self, user_id: str, skill_id_or_name: str) -> bool:
        """Reactivate a previously disabled skill."""
        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            return False
        skill.is_active = True
        self.repository.update(skill)
        return True

    def update_skill(
        self,
        user_id: str,
        skill_id_or_name: str,
        name: str | None = None,
        description: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        parameters_schema: dict[str, Any] | None = None,
    ) -> Skill | None:
        """Update an existing skill with a new version."""
        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            return None

        new_name = name or skill.name
        new_desc = description or skill.description
        new_steps = steps or skill.steps
        new_params = parameters_schema if parameters_schema is not None else skill.parameters_schema

        validate_skill_security(new_name, new_desc, new_steps, new_params)

        skill.name = new_name
        skill.description = new_desc
        skill.steps = new_steps
        skill.parameters_schema = new_params
        skill.version += 1
        return self.repository.update(skill)

    def delete_skill(self, user_id: str, skill_id_or_name: str) -> bool:
        """Delete a skill belonging to the user."""
        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            return False
        return self.repository.delete(user_id, skill.id)

    def adapt_skill_with_llm(
        self,
        user_id: str,
        skill_id_or_name: str,
        new_user_goal: str,
    ) -> AdaptedPlanModel:
        """
        Adapt an approved saved skill to a new user request using Nemotron assistant (P16.1).
        """
        if not self.nemotron_assistant:
            raise SkillValidationError("Nemotron assistant is not configured for skill adaptation.")

        skill = self.get_skill(user_id, skill_id_or_name)
        if not skill:
            raise SkillValidationError(f"Skill '{skill_id_or_name}' not found for user '{user_id}'.")

        skill_dict = {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "steps": skill.steps,
            "parameters_schema": skill.parameters_schema,
        }
        return self.nemotron_assistant.adapt_skill_for_goal(skill_dict, new_user_goal)

    def summarize_workflow_outcome(
        self,
        skill_name: str,
        task_goal: str,
        step_results: list[dict[str, Any]],
    ) -> WorkflowSummaryModel:
        """
        Summarize a workflow outcome into reusable, non-sensitive metadata using Nemotron (P16.1).
        """
        if self.nemotron_assistant:
            return self.nemotron_assistant.summarize_workflow_outcome(skill_name, task_goal, step_results)

        return WorkflowSummaryModel(
            summary=f"Successfully executed skill '{skill_name}' across {len(step_results)} steps.",
            reusable_insights=["Completed deterministically"],
            recommended_parameters=[],
        )

