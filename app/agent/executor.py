"""
P6 — Safe Plan Executor.
Executes an ExecutionPlan with strict safety constraints:
- Hard step limit (MAX_PLAN_STEPS)
- Loop / repeated call protection
- Tool registry existence validation
- Argument schema validation
- Dependency checking and skipping
- Controlled intermediate result passing
- Safe error handling (never crash)
"""
from datetime import datetime, timedelta
import json
import re
from typing import Any

from app.agent.capabilities import ALLOWED_TOOL_DEPENDENCIES, get_tool_capability
from app.agent.permissions import PermissionManager
from app.agent.plan import ExecutionPlan, StepResult, ToolStep, MAX_PLAN_STEPS, KNOWN_EXTRACTORS
from app.agent.trace import ExecutionTrace
from app.core.config import settings
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry
from app.tools.web.ranking import rank_search_results
from app.tools.web.url import deduplicate_search_results


class PlanExecutor:
    """Executes multi-step agent plans deterministically with safety boundaries."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager | None = None,
    ):
        self.tool_registry = tool_registry
        self.permission_manager = permission_manager or PermissionManager()

    def validate_plan(
        self,
        plan: ExecutionPlan,
        user_message: str = "",
    ) -> tuple[bool, str | None]:
        """
        Validate the structural integrity and constraints of an ExecutionPlan before execution.
        
        Checks:
        1. Max step limit (len <= MAX_PLAN_STEPS)
        2. Unique step IDs
        3. All tool names exist in tool_registry
        4. All argument extractors are known in KNOWN_EXTRACTORS
        5. All dependencies exist and precede dependent steps (no forward references or self-dependencies)
        6. No dependency cycles
        7. Supported tool combinations and extractors
        8. Memory mutation authorization (only with explicit user intent)
        9. Required arguments and argument types for steps without extractors
        """
        if len(plan.steps) > MAX_PLAN_STEPS:
            return False, f"Plan step limit exceeded: {len(plan.steps)} steps requested, maximum is {MAX_PLAN_STEPS}."

        seen_ids: set[str] = set()
        step_id_to_index: dict[str, int] = {}
        for idx, step in enumerate(plan.steps):
            if step.step_id in seen_ids:
                return False, f"Duplicate step ID detected: '{step.step_id}'."
            seen_ids.add(step.step_id)
            step_id_to_index[step.step_id] = idx

        for step in plan.steps:
            # 1. Tool existence check
            is_memory_tool = step.tool_name in {"remember_memory", "search_memory", "update_memory", "forget_memory"}
            if not (self.tool_registry.has(step.tool_name) or is_memory_tool):
                return False, f"The requested tool '{step.tool_name}' is not available."

            # 2. Extractor validity check
            if step.extractor and step.extractor not in KNOWN_EXTRACTORS:
                return False, f"Unknown argument extractor: '{step.extractor}'."

            # 3. Dependency existence and ordering
            for dep in step.depends_on:
                if dep not in seen_ids:
                    return False, f"Unknown dependency: step '{step.step_id}' depends on non-existent step '{dep}'."
                if dep == step.step_id:
                    return False, f"Self-dependency detected: step '{step.step_id}' cannot depend on itself."
                if step_id_to_index[dep] >= step_id_to_index[step.step_id]:
                    return False, f"Dependency cycle or invalid forward ordering: step '{step.step_id}' depends on step '{dep}' which does not precede it."

                # 4. Supported tool combination check
                dep_step = plan.get_step(dep)
                if dep_step:
                    pair = (dep_step.tool_name, step.tool_name)
                    if pair in ALLOWED_TOOL_DEPENDENCIES:
                        allowed_exts = ALLOWED_TOOL_DEPENDENCIES[pair]
                        if step.extractor not in allowed_exts:
                            return False, (
                                f"Invalid extractor '{step.extractor}' for dependency from "
                                f"'{dep_step.tool_name}' to '{step.tool_name}'. Allowed: {allowed_exts}."
                            )
                    else:
                        known_std = {
                            "calculator", "datetime", "web_search", "web_fetch",
                            "file_search", "remember_memory", "search_memory",
                            "update_memory", "forget_memory",
                        }
                        if dep_step.tool_name in known_std and step.tool_name in known_std:
                            return False, (
                                f"Unsupported tool combination: step '{step.step_id}' ({step.tool_name}) "
                                f"cannot depend on step '{dep}' ({dep_step.tool_name})."
                            )

            # 5. Memory authorization check
            memory_mutation_tools = {"remember_memory", "update_memory", "forget_memory"}
            if step.tool_name in memory_mutation_tools:
                has_explicit_memory_intent = False
                if user_message:
                    low = user_message.lower()
                    has_explicit_memory_intent = bool(
                        re.search(r"\b(remember|forget|save to memory|update memory|note that)\b", low)
                    )
                if not has_explicit_memory_intent:
                    return False, f"Unauthorized memory mutation: tool '{step.tool_name}' requires explicit user instruction."

            # 6. Required arguments and argument types validation (for steps without dynamic extractors)
            if not step.extractor and self.tool_registry.has(step.tool_name):
                tool = self.tool_registry.get(step.tool_name)
                if hasattr(tool, "input_schema") and tool.input_schema is not None:
                    fields = getattr(tool.input_schema, "model_fields", {})
                    for field_name, field_info in fields.items():
                        if field_info.is_required() and field_name not in step.arguments:
                            return False, f"Missing required argument '{field_name}' for tool '{step.tool_name}'."
                    try:
                        tool.input_schema.model_validate(step.arguments)
                    except Exception as val_e:
                        return False, f"Invalid arguments for tool '{step.tool_name}': {val_e}"

        # 7. Explicit cycle detection via DFS
        adj: dict[str, list[str]] = {s.step_id: list(s.depends_on) for s in plan.steps}
        visited: dict[str, int] = {s.step_id: 0 for s in plan.steps}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node: str) -> str | None:
            visited[node] = 1
            for neighbor in adj.get(node, []):
                if visited.get(neighbor) == 1:
                    return f"Dependency cycle detected involving step '{neighbor}'."
                if visited.get(neighbor) == 0:
                    err = dfs(neighbor)
                    if err:
                        return err
            visited[node] = 2
            return None

        for s in plan.steps:
            if visited[s.step_id] == 0:
                cycle_err = dfs(s.step_id)
                if cycle_err:
                    return False, cycle_err

        return True, None

    def execute(
        self,
        plan: ExecutionPlan,
        user_message: str,
        user_id: str | None = None,
        trace: ExecutionTrace | None = None,
    ) -> list[StepResult]:
        """Execute all steps in an execution plan according to dependencies and safety limits."""
        if trace:
            trace.plan_id = plan.plan_id
            trace.user_message = user_message
            trace.candidate_plan = {
                "plan_id": plan.plan_id,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "tool_name": s.tool_name,
                        "arguments": s.arguments,
                        "depends_on": s.depends_on,
                        "extractor": s.extractor,
                        "purpose": s.purpose,
                    }
                    for s in plan.steps
                ],
            }

        is_valid, validation_error = self.validate_plan(plan, user_message=user_message)
        if not is_valid:
            if trace:
                trace.validation_passed = False
                trace.validation_error = validation_error
                trace.final_status = "validation_failed"
            step_id = "safety_limit" if "limit exceeded" in (validation_error or "").lower() else "plan_validation"
            return [
                StepResult(
                    step_id=step_id,
                    tool_name="system",
                    success=False,
                    error=validation_error,
                    output_text=f"Plan validation failed: {validation_error}",
                )
            ]

        if trace:
            trace.validation_passed = True

        results: list[StepResult] = []
        completed: dict[str, StepResult] = {}
        executed_calls: set[str] = set()

        for step in plan.steps:
            # 2. Dependency resolution check
            if step.depends_on:
                prereqs_ok = True
                failed_dep = ""
                for dep_id in step.depends_on:
                    dep_res = completed.get(dep_id)
                    if not dep_res or not dep_res.success or dep_res.skipped:
                        prereqs_ok = False
                        failed_dep = dep_id
                        break

                if not prereqs_ok:
                    skip_res = StepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        skipped=True,
                        skip_reason=f"Prerequisite step '{failed_dep}' did not succeed; dependent step skipped.",
                        output_text=f"Dependent step '{step.step_id}' ({step.tool_name}) was skipped because prerequisite '{failed_dep}' did not succeed.",
                    )
                    results.append(skip_res)
                    completed[step.step_id] = skip_res
                    continue

            # 3. Dynamic argument resolution from dependencies
            step_args = dict(step.arguments)
            if step.extractor:
                extraction_ok, extraction_err = self._resolve_extracted_arguments(
                    step=step,
                    step_args=step_args,
                    completed=completed,
                    user_message=user_message,
                )
                if not extraction_ok:
                    skip_reason = extraction_err or f"Could not extract necessary data from prerequisite for {step.tool_name}."
                    skip_res = StepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        skipped=True,
                        skip_reason=skip_reason,
                        output_text=f"Dependent step '{step.step_id}' ({step.tool_name}) was skipped: {skip_reason}",
                    )
                    results.append(skip_res)
                    completed[step.step_id] = skip_res
                    continue

            # 4. Repeated call / loop protection
            # Normalize arguments for comparison, excluding dynamic runtime keys
            normalized_args_dict = {
                k: str(v)
                for k, v in step_args.items()
                if k not in ("user_message", "candidate_pool_size", "confirmed")
            }
            normalized_args_repr = json.dumps(normalized_args_dict, sort_keys=True)
            call_sig = f"{step.tool_name}:{normalized_args_repr}"
            if call_sig in executed_calls:
                loop_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Loop protection: duplicate call to '{step.tool_name}' with identical arguments prevented.",
                    output_text=f"Repeated execution loop prevented for tool '{step.tool_name}'.",
                )
                results.append(loop_res)
                completed[step.step_id] = loop_res
                break
            executed_calls.add(call_sig)

            # 5. Permission check (guards write/mutating actions before execution or tool availability)
            permission = self.permission_manager.check(
                tool_name=step.tool_name,
                user_id=user_id,
                confirmed=bool(step_args.get("confirmed")),
            )
            if permission.requires_confirmation:
                perm_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Confirmation required for tool '{step.tool_name}'.",
                    output_text=f"I need your confirmation before I {step.tool_name.replace('_', ' ')}. Please confirm that you want me to proceed.",
                )
                results.append(perm_res)
                completed[step.step_id] = perm_res
                break

            # 6. Tool existence check
            if not self.tool_registry.has(step.tool_name):
                err_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"The requested tool '{step.tool_name}' is not available.",
                    output_text=f"Tool '{step.tool_name}' is not registered.",
                )
                results.append(err_res)
                completed[step.step_id] = err_res
                continue

            tool = self.tool_registry.get(step.tool_name)

            # 7. User ID assignment for user-scoped tools (memory, etc.)
            if user_id and hasattr(tool, "set_user_id"):
                tool.set_user_id(user_id)

            # 8. Web search specific argument defaults
            if step.tool_name == "web_search":
                candidate_pool = getattr(settings, "web_search_candidate_pool_size", 10)
                step_args["candidate_pool_size"] = candidate_pool

            # 9. Tool argument validation
            try:
                if hasattr(tool, "input_schema") and tool.input_schema is not None:
                    # Validate schema
                    validated_model = tool.input_schema.model_validate(step_args)
                    # Pass validated dump or dictionary
                    clean_kwargs = validated_model.model_dump(exclude_unset=False)
                    # Preserve candidate_pool_size if present
                    if "candidate_pool_size" in step_args:
                        clean_kwargs["candidate_pool_size"] = step_args["candidate_pool_size"]
                else:
                    clean_kwargs = step_args
            except Exception as val_err:
                val_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Invalid arguments for tool '{step.tool_name}': {val_err}",
                    output_text=f"Validation failed for {step.tool_name} arguments.",
                )
                results.append(val_res)
                completed[step.step_id] = val_res
                continue

            # 10. Execute tool safely
            try:
                tool_result = tool.run(**clean_kwargs)
                formatted_text = self._format_tool_result(tool_result, user_message=user_message)
                step_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=tool_result.success,
                    data=tool_result.data,
                    arguments=dict(step_args),
                    error=tool_result.error,
                    output_text=formatted_text,
                )
            except Exception as exec_err:
                step_res = StepResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    arguments=dict(step_args),
                    error=str(exec_err),
                    output_text=f"The {step.tool_name} tool failed: {exec_err}",
                )

            results.append(step_res)
            completed[step.step_id] = step_res

        if trace:
            trace.executed_steps = [r.step_id for r in results if not r.skipped]
            trace.step_results = [r.to_compact_dict() for r in results]
            trace.skipped_steps = [r.step_id for r in results if r.skipped]
            trace.failure_reasons = [r.skip_reason or r.error or "failed" for r in results if not r.success]
            if all(r.success for r in results):
                trace.final_status = "success"
            elif any(r.success for r in results):
                trace.final_status = "partial"
            else:
                trace.final_status = "failed"

        return results

    def _resolve_extracted_arguments(
        self,
        step: ToolStep,
        step_args: dict[str, Any],
        completed: dict[str, StepResult],
        user_message: str,
    ) -> tuple[bool, str | None]:
        """Resolve dynamic arguments for a step from previous step results."""
        dep_id = step.depends_on[0] if step.depends_on else "step_1"
        dep_res = completed.get(dep_id)

        if step.extractor == "extract_version_comparison":
            if not dep_res or not dep_res.success:
                return False, f"Prerequisite step '{dep_id}' did not succeed."

            raw_text = dep_res.output_text or ""
            if dep_res.data and isinstance(dep_res.data, dict):
                snippets = " ".join(
                    r.get("title", "") + " " + r.get("snippet", "")
                    for r in dep_res.data.get("results", [])
                )
                raw_text += " " + snippets
                page_content = (
                    (dep_res.data.get("title") or "")
                    + " "
                    + (dep_res.data.get("content") or dep_res.data.get("text") or "")
                )
                if page_content.strip():
                    raw_text += " " + page_content

            entity = str(step_args.get("entity", "python")).lower()
            target_str = str(step_args.get("target_version", "3.13"))

            # Entity-isolated version search:
            # 1. Look for entity name followed by version within 40 characters
            entity_match = re.search(
                rf"\b{re.escape(entity)}\b[^\d\n]{{0,40}}([vV]?\d+(?:\.\d+)+)",
                raw_text,
                re.IGNORECASE,
            )
            discovered_version: str | None = None
            if entity_match:
                discovered_version = entity_match.group(1).lstrip("vV")
            else:
                # 2. Look for lines/snippets mentioning the entity
                entity_lines = [
                    line for line in raw_text.splitlines()
                    if entity in line.lower()
                ]
                for line in entity_lines:
                    vm = re.search(r"\b([vV]?\d+(?:\.\d+)+)\b", line)
                    if vm:
                        discovered_version = vm.group(1).lstrip("vV")
                        break

            # 3. Entity-specific fallbacks if not yet found
            if not discovered_version:
                if entity == "python" and "python" in raw_text.lower():
                    vm = re.search(r"\b3\.(\d+)(?:\.(\d+))?\b", raw_text)
                    if vm:
                        discovered_version = vm.group(0)
                elif entity == "fastapi" and "fastapi" in raw_text.lower():
                    vm = re.search(r"\b0\.(\d+)(?:\.(\d+))?\b", raw_text)
                    if vm:
                        discovered_version = vm.group(0)
                elif entity == "cuda" and "cuda" in raw_text.lower():
                    vm = re.search(r"\b12\.(\d+)(?:\.(\d+))?\b", raw_text)
                    if vm:
                        discovered_version = vm.group(0)

            if not discovered_version:
                return False, f"Could not extract necessary data from prerequisite: could not discover version for entity '{entity}' in search results."

            # Parse discovered version and target version
            def parse_version_parts(v: str) -> tuple[int, int] | None:
                clean = re.sub(r"^[vV]", "", v.strip())
                nums = [int(p) for p in re.findall(r"\d+", clean)]
                if len(nums) >= 2:
                    return nums[0], nums[1]
                if len(nums) == 1:
                    return nums[0], 0
                return None

            disc_parsed = parse_version_parts(discovered_version)
            target_parsed = parse_version_parts(target_str)

            if not disc_parsed or not target_parsed:
                return False, (
                    f"Incompatible version format: could not parse major.minor from "
                    f"discovered '{discovered_version}' or target '{target_str}'."
                )

            disc_major, disc_minor = disc_parsed
            target_major, target_minor = target_parsed

            # Enforce major-version compatibility: minor-version comparison requires same major version
            if disc_major != target_major:
                return False, (
                    f"Incompatible major versions for minor-version comparison: "
                    f"discovered major version {disc_major} != target major version {target_major}."
                )

            diff = disc_minor - target_minor
            step_args["expression"] = f"{disc_minor} - {target_minor}"
            step_args["discovered_version"] = discovered_version
            step_args["target_version"] = target_str
            step_args["minor_difference"] = diff
            step_args["entity"] = entity
            step_args["major_version"] = disc_major
            return True, None

        if step.extractor == "extract_date_offset":
            if not dep_res or not dep_res.success or not dep_res.data:
                return False, "Prerequisite datetime step failed or missing date."

            ref_date_str = dep_res.data.get("date")
            if not ref_date_str:
                return False, "Reference date missing in datetime result."

            days_val = step_args.get("days")
            if days_val is None:
                days_val = step_args.get("offset", 10)

            try:
                days_offset = int(days_val)
            except (ValueError, TypeError):
                return False, f"Invalid date offset: '{days_val}' is not a valid integer."

            try:
                ref_dt = datetime.strptime(ref_date_str, "%Y-%m-%d")
                target_dt = ref_dt + timedelta(days=days_offset)
                target_str = target_dt.strftime("%Y-%m-%d")
                target_day = target_dt.strftime("%A")

                step_args["expression"] = f"{days_offset} + 0" if days_offset >= 0 else f"{days_offset}"
                step_args["calculated_date"] = target_str
                step_args["calculated_day"] = target_day
                step_args["days"] = days_offset
                return True, None
            except Exception as e:
                return False, f"Failed calculating date offset: {e}"

        if step.extractor == "extract_search_url":
            if not dep_res or not dep_res.success:
                return False, f"Prerequisite step '{dep_id}' did not succeed."

            extracted_url = None
            if dep_res.data and isinstance(dep_res.data, dict):
                results_list = dep_res.data.get("results", [])
                if results_list and isinstance(results_list, list):
                    first_item = results_list[0]
                    if isinstance(first_item, dict) and "url" in first_item:
                        extracted_url = first_item["url"]

            if not extracted_url:
                url_matches = re.findall(r"https?://[^\s)\]]+", dep_res.output_text or "")
                if url_matches:
                    extracted_url = url_matches[0]

            if not extracted_url:
                return False, f"Could not extract search URL from prerequisite step '{dep_id}'."

            step_args["url"] = extracted_url
            return True, None

        if step.extractor == "extract_date_for_search":
            if not dep_res or not dep_res.success:
                return False, f"Prerequisite step '{dep_id}' did not succeed."

            date_val = None
            if dep_res.data and isinstance(dep_res.data, dict):
                date_val = dep_res.data.get("date")
            if not date_val:
                dm = re.search(r"\b\d{4}-\d{2}-\d{2}\b", dep_res.output_text or "")
                if dm:
                    date_val = dm.group(0)

            if not date_val:
                return False, f"Could not extract reference date from prerequisite step '{dep_id}'."

            current_query = step_args.get("query", "").strip()
            if date_val not in current_query:
                step_args["query"] = f"{current_query} {date_val}".strip()
            return True, None

        if step.extractor == "extract_page_calculation":
            if not dep_res or not dep_res.success:
                return False, f"Prerequisite step '{dep_id}' did not succeed."

            raw_text = ""
            if dep_res.data and isinstance(dep_res.data, dict):
                raw_text = dep_res.data.get("content") or dep_res.data.get("text") or ""
            if not raw_text:
                raw_text = dep_res.output_text or ""

            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", raw_text)
            if not numbers:
                return False, f"Could not extract numerical values from page content in step '{dep_id}'."
            if "expression" not in step_args or not step_args["expression"]:
                if len(numbers) >= 2:
                    step_args["expression"] = f"{numbers[0]} + {numbers[1]}"
                else:
                    step_args["expression"] = numbers[0]
            return True, None

        if step.extractor == "extract_chained_calculation":
            if not dep_res or not dep_res.success:
                return False, f"Prerequisite step '{dep_id}' did not succeed."

            prev_num = None
            if dep_res.data is not None and isinstance(dep_res.data, (int, float)):
                prev_num = dep_res.data
            elif dep_res.data and isinstance(dep_res.data, dict) and "result" in dep_res.data:
                prev_num = dep_res.data["result"]
            else:
                m = re.search(r"\b(-?\d+(?:\.\d+)?)\b", dep_res.output_text)
                if m:
                    prev_num = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

            if prev_num is None:
                return False, f"Could not extract numeric value from step '{dep_id}'."

            op = step_args.get("operation", "/")
            operand = step_args.get("operand", 1)
            step_args["expression"] = f"{prev_num} {op} {operand}"
            return True, None

        return False, f"Unknown argument extractor: '{step.extractor}'."


    def _format_tool_result(self, result, user_message: str = "") -> str:
        """Format a ToolResult into a clear, standardized text representation."""
        if not result.success:
            return (
                f"The {result.tool_name} tool could not complete the request. "
                f"Reason: {result.error or 'unknown error'}"
            )

        if result.tool_name == "calculator":
            val = (result.data or {}).get("result")
            return f"The calculation result is {val}."

        if result.tool_name == "datetime":
            d = result.data or {}
            date = d.get("date", "")
            time_str = d.get("time", "")
            day = d.get("day", "")
            tz = d.get("timezone", "")
            loc = d.get("location")
            qtype = d.get("query_type")
            rel = d.get("relative_day", "today")

            loc_str = f" in {loc}" if loc else ""
            if rel == "tomorrow":
                if qtype == "day":
                    return f"Tomorrow will be {day}."
                if qtype == "date":
                    return f"Tomorrow's date will be {date}."
                return f"Tomorrow will be {day}, {date} ({tz})."
            if rel == "yesterday":
                if qtype == "day":
                    return f"Yesterday was {day}."
                if qtype == "date":
                    return f"Yesterday's date was {date}."
                return f"Yesterday was {day}, {date} ({tz})."
            if qtype == "time":
                return f"The current time{loc_str} is {time_str} ({tz})."
            if qtype == "date":
                return f"Today's date{loc_str} is {date} ({tz})."
            if qtype == "day":
                return f"Today{loc_str} is {day}."
            return (
                f"Today is {day}, {date}, "
                f"and the current time{loc_str} is {time_str} ({tz})."
            )

        if result.tool_name == "remember_memory":
            return "I saved that to your long-term memory."

        if result.tool_name == "search_memory":
            memories = (result.data or {}).get("memories", [])
            if not memories:
                return "I couldn't find a matching memory."
            details = "; ".join(item.get("content", "") for item in memories)
            return f"I remember: {details}."

        if result.tool_name == "web_search":
            data = result.data or {}
            raw_results = data.get("results", [])
            raw_answer = data.get("answer")

            clean_answer = ""
            if raw_answer is not None:
                try:
                    if isinstance(raw_answer, str) and raw_answer.strip():
                        clean_answer = raw_answer.strip()
                    elif isinstance(raw_answer, (int, float)) and str(raw_answer).strip():
                        clean_answer = str(raw_answer).strip()
                except Exception:
                    clean_answer = ""

            max_limit = data.get("max_results") or getattr(settings, "web_search_max_results", 5)
            search_query = data.get("query") or user_message
            freshness = data.get("freshness")
            domain_restriction = data.get("domain")

            deduped_sources = deduplicate_search_results(raw_results, max_results=None)
            valid_sources = rank_search_results(
                deduped_sources,
                query=search_query,
                freshness=freshness,
                domain_restriction=domain_restriction,
                max_results=max_limit,
            )

            if not valid_sources and not clean_answer:
                return "I couldn't find recent web information for that query."

            blocks = []
            if clean_answer:
                blocks.append(f"<Tavily_answer>\n{clean_answer}\n</Tavily_answer>")

            for index, source in enumerate(valid_sources, start=1):
                blocks.append(
                    f"<SOURCE_{index}>\n"
                    f"Title: {source['title']}\n"
                    f"URL: {source['url']}\n"
                    f"Snippet: {source['snippet']}\n"
                    f"</SOURCE_{index}>"
                )

            return "\n\n".join(blocks).strip()

        if result.tool_name == "web_fetch":
            data = result.data or {}
            url = data.get("url") or ""
            final_url = data.get("final_url") or url
            title = data.get("title") or "Web Page"
            text = data.get("text") or data.get("content") or ""
            truncated = data.get("truncated", False)

            trunc_notice = " [Content truncated for length]" if truncated else ""
            return (
                f"<page_evidence>\n"
                f"<PAGE_SOURCE>\n"
                f"URL: {url}\n"
                f"Final URL: {final_url}\n"
                f"Title: {title}\n"
                f"Content:\n{text}{trunc_notice}\n"
                f"</PAGE_SOURCE>\n"
                f"</page_evidence>"
            )

        if result.tool_name == "file_search":
            matches = (result.data or {}).get("matches", []) or (result.data or {}).get("results", [])
            if not matches:
                return "I couldn't find a relevant file match for that request."

            lines = [f"I found these relevant file matches for '{result.data.get('query', '')}':"]
            for index, item in enumerate(matches[:3], start=1):
                title = item.get("title", "Document")
                path = item.get("path", "")
                snippet = item.get("snippet", "")
                if snippet:
                    lines.append(f"{index}. {title} — {snippet} ({path})")
                else:
                    lines.append(f"{index}. {title} ({path})")
            return "\n".join(lines)

        if result.tool_name == "update_memory":
            if result.success:
                return "I updated that memory."
            return "I couldn't find that memory to update."

        if result.tool_name == "forget_memory":
            if (result.data or {}).get("deleted"):
                return "I forgot that memory."
            return "I couldn't find a matching memory to forget."

        return str(result.data)
