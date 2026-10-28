"""
P16 — Skill Execution Engine.
Validates inputs, substitutes parameters, enforces tool registry integrity,
and executes saved procedures through the established PlanExecutor.
"""
from datetime import datetime, timezone
import time
from typing import Any

from app.agent.executor import PlanExecutor
from app.agent.plan import ExecutionPlan, StepResult, ToolStep
from app.agent.task_state import ActiveTaskState
from app.database.models import Skill
from app.database.repositories.skills import SkillRepository
from app.skills.model import SkillExecutionResult, SkillValidationError
from app.tools.registry import ToolRegistry


class SkillExecutor:
    """Safely executes validated procedural skills with parameter substitution and tool verification."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        skill_repo: SkillRepository | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.skill_repo = skill_repo
        self.plan_executor = PlanExecutor(tool_registry=tool_registry)

    def validate_tools_available(self, skill: Skill) -> tuple[bool, str | None]:
        """Verify that every tool referenced by the skill currently exists in the registry."""
        for step in skill.steps:
            t_name = step.get("tool_name", "")
            is_memory_tool = t_name in {"remember_memory", "search_memory", "update_memory", "forget_memory"}
            if not (self.tool_registry.has(t_name) or is_memory_tool):
                return False, f"Tool '{t_name}' required by skill '{skill.name}' is not registered."
        return True, None

    def substitute_parameters(
        self,
        step_args: dict[str, Any],
        validated_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute validated parameters into step argument templates."""
        substituted: dict[str, Any] = {}
        for k, v in step_args.items():
            if isinstance(v, str):
                # Replace exact placeholder e.g. "{query}"
                val = v
                for param_name, param_val in validated_inputs.items():
                    placeholder = f"{{{param_name}}}"
                    if val == placeholder:
                        val = param_val
                        break
                    elif placeholder in val:
                        val = val.replace(placeholder, str(param_val))
                substituted[k] = val
            elif isinstance(v, dict):
                substituted[k] = self.substitute_parameters(v, validated_inputs)
            else:
                substituted[k] = v
        return substituted

    def validate_and_filter_inputs(
        self,
        skill: Skill,
        user_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate that required parameters are present, and filter to ONLY
        the parameters the skill is designed to accept.
        """
        schema = skill.parameters_schema or {}
        filtered: dict[str, Any] = {}

        for param_name, p_spec in schema.items():
            is_required = p_spec.get("required", True) if isinstance(p_spec, dict) else True
            if param_name in user_inputs:
                filtered[param_name] = user_inputs[param_name]
            elif is_required:
                raise SkillValidationError(
                    f"Missing required parameter '{param_name}' for skill '{skill.name}'."
                )

        return filtered

    def build_execution_plan(
        self,
        skill: Skill,
        validated_inputs: dict[str, Any],
    ) -> ExecutionPlan:
        """Construct an ExecutionPlan from the skill's steps with substituted arguments."""
        tool_steps: list[ToolStep] = []
        for step_dict in skill.steps:
            raw_args = step_dict.get("arguments", {})
            sub_args = self.substitute_parameters(raw_args, validated_inputs)
            tool_steps.append(
                ToolStep(
                    step_id=step_dict.get("step_id", f"step_{len(tool_steps)+1}"),
                    tool_name=step_dict["tool_name"],
                    arguments=sub_args,
                    depends_on=list(step_dict.get("depends_on", [])),
                    purpose=step_dict.get("purpose", ""),
                    extractor=step_dict.get("extractor"),
                )
            )

        return ExecutionPlan(
            plan_id=f"plan_{skill.id[:12]}",
            steps=tool_steps,
            final_response_required=True,
            context={"skill_id": skill.id, "skill_name": skill.name},
        )

    def execute(
        self,
        skill: Skill,
        user_inputs: dict[str, Any],
        active_task: ActiveTaskState | None = None,
        user_id: str = "default",
    ) -> SkillExecutionResult:
        """
        Execute a saved skill end-to-end:
        1. Tool availability check (mark needs_repair if tool missing)
        2. Parameter validation & filtering
        3. Plan construction & structural validation
        4. Plan execution through PlanExecutor
        5. Record outcome and return result
        """
        t0 = time.perf_counter()

        # 1. Tool availability verification
        tools_ok, tool_err = self.validate_tools_available(skill)
        if not tools_ok:
            if not skill.needs_repair:
                skill.needs_repair = True
                if self.skill_repo:
                    self.skill_repo.update(skill)
            return SkillExecutionResult(
                skill_id=skill.id,
                skill_name=skill.name,
                success=False,
                steps_executed=0,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                output_summary=f"Skill '{skill.name}' needs repair: {tool_err}",
                error=tool_err,
            )

        # 2. Parameter validation
        try:
            validated_inputs = self.validate_and_filter_inputs(skill, user_inputs)
        except SkillValidationError as e:
            return SkillExecutionResult(
                skill_id=skill.id,
                skill_name=skill.name,
                success=False,
                steps_executed=0,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                output_summary=str(e),
                error=str(e),
            )

        # 3. Plan construction & validation
        plan = self.build_execution_plan(skill, validated_inputs)
        is_plan_valid, plan_err = self.plan_executor.validate_plan(plan)
        if not is_plan_valid:
            return SkillExecutionResult(
                skill_id=skill.id,
                skill_name=skill.name,
                success=False,
                steps_executed=0,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                output_summary=f"Skill execution rejected: {plan_err}",
                error=plan_err,
            )

        # 4. Execute plan through existing executor
        step_results: list[StepResult] = self.plan_executor.execute(
            plan=plan,
            user_message=f"Execute skill {skill.name}",
            user_id=user_id,
        )

        duration_ms = (time.perf_counter() - t0) * 1000
        all_success = all(r.success for r in step_results) and len(step_results) == len(plan.steps)
        failed_step = next((r for r in step_results if not r.success), None)
        failed_step_index = next((idx for idx, r in enumerate(step_results) if not r.success), None)
        err_msg = failed_step.error if failed_step else None

        # Synthesize concise explanation
        summaries = []
        intermediate: dict[str, Any] = {}
        for r in step_results:
            step_summary = r.output_text or str(r.data or "") or ("OK" if r.success else (r.error or "Failed"))
            summaries.append(f"{r.tool_name}: {step_summary}")
            if r.data is not None:
                intermediate[r.step_id] = r.data

        summary_text = f"Executed skill '{skill.name}': " + "; ".join(summaries)

        # 5. Record outcome in repository
        if self.skill_repo:
            self.skill_repo.record_outcome(
                user_id=user_id,
                skill_id=skill.id,
                success=all_success,
                error=err_msg,
                steps_executed=len(step_results),
                duration_ms=duration_ms,
            )

        return SkillExecutionResult(
            skill_id=skill.id,
            skill_name=skill.name,
            success=all_success,
            steps_executed=len(step_results),
            duration_ms=round(duration_ms, 2),
            output_summary=summary_text,
            error=err_msg,
            failed_step_index=failed_step_index,
            intermediate_data=intermediate,
        )
