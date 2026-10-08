"""
P6 — Deterministic Execution Planner.
Constructs an explicit, inspectable ExecutionPlan from the user message and routing decision.
"""
import re
from typing import Any
import uuid

from app.agent.plan import ExecutionPlan, ToolStep, MAX_PLAN_STEPS
from app.agent.tool_argument_resolver import (
    resolve_calculator_arguments,
    resolve_datetime_arguments,
    resolve_web_fetch_arguments,
    resolve_web_search_arguments,
)
from app.schemas.agent import AgentDecision


def extract_entity_and_target_version(user_message: str) -> tuple[str, str]:
    """
    Extract the software entity name (e.g. 'python', 'fastapi', 'cuda')
    and comparison target version from the user's message.
    """
    lowered = user_message.lower()

    # 1. Detect target entity
    entity = "python"
    if "fastapi" in lowered:
        entity = "fastapi"
    elif "cuda" in lowered:
        entity = "cuda"
    elif "python" in lowered:
        entity = "python"
    else:
        m = re.search(r"(?:for|about)\s+(?:the\s+)?(?:latest\s+)?([a-zA-Z0-9_-]+)\s+(?:release|version)", user_message, re.IGNORECASE)
        if m:
            entity = m.group(1).lower()

    # 2. Detect target comparison version
    target_match = re.search(
        r"(?:compare\s+(?:it\s+)?(?:with|to)|newer\s+(?:it\s+is\s+)?than|older\s+(?:it\s+is\s+)?than|than)\s+(?:[A-Za-z0-9_-]+\s*)?([vV]?\d+(?:\.\d+)+)",
        user_message,
        re.IGNORECASE,
    )
    if target_match:
        target_version = target_match.group(1).lstrip("vV")
    else:
        # Fallback to any version pattern near comparison keywords
        v_m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", user_message)
        target_version = v_m.group(1) if v_m else "3.13"

    return entity, target_version


def build_execution_plan(
    user_message: str,
    decision: AgentDecision,
    user_id: str | None = None,
    active_task: Any = None,
) -> ExecutionPlan:
    """Build a deterministic, inspectable ExecutionPlan."""
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    lowered = user_message.lower().strip()

    # 0a. Check active task URL reuse (prevent redundant web search)
    has_task_url = bool(
        active_task
        and not active_task.is_expired()
        and active_task.context_values.get("selected_url")
    )
    is_task_url_reference = any(
        kw in lowered
        for kw in (
            "that page",
            "the same page",
            "summarize that page",
            "summarize the page",
            "summarize the release notes",
            "summarize release notes",
            "read the release notes",
            "summarize it",
        )
    )
    url_match = re.search(r"https?://[^\s>]+", user_message)
    if has_task_url and is_task_url_reference and not url_match:
        task_url = active_task.context_values["selected_url"]
        fetch_args = resolve_web_fetch_arguments(user_message, {"url": task_url})
        step = ToolStep(
            step_id="step_1",
            tool_name="web_fetch",
            arguments=fetch_args,
            depends_on=[],
            purpose=f"Fetch page content from active task URL: {task_url}",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step])

    # 0b. Chained arithmetic calculation
    # e.g. "Calculate 20 * 5 and then divide that result by 2."
    chained_calc_match = re.search(
        r"(?:calculate\s+)?(\d+\s*[\+\-\*/%]\s*\d+)\s+and\s+(?:then\s+)?(divide|multiply|add|subtract|plus|minus|times)\s+(?:that\s+result|the\s+result|it)\s+(?:by|with|to|from)\s+(\d+(?:\.\d+)?)",
        lowered,
    )
    if chained_calc_match:
        first_expr = chained_calc_match.group(1)
        op_word = chained_calc_match.group(2)
        operand_str = chained_calc_match.group(3)
        operand = float(operand_str) if "." in operand_str else int(operand_str)
        op_map = {
            "divide": "/",
            "multiply": "*",
            "times": "*",
            "add": "+",
            "plus": "+",
            "subtract": "-",
            "minus": "-",
        }
        op = op_map.get(op_word, "/")
        step_1 = ToolStep(
            step_id="step_1",
            tool_name="calculator",
            arguments={"expression": first_expr},
            depends_on=[],
            purpose=f"Calculate {first_expr}",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="calculator",
            arguments={"operation": op, "operand": operand},
            depends_on=["step_1"],
            purpose=f"{op_word.capitalize()} previous result by {operand}",
            extractor="extract_chained_calculation",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 1. Detect Web Search + Compare / Calculate workflow
    # e.g. "Search the web for the latest Python release and compare it with Python 3.13."
    # e.g. "Search the web for the latest Python version and tell me how many minor versions newer it is than 3.12."
    # e.g. "Search for the latest FastAPI release and compare it with 0.110."
    # e.g. "Search for the latest CUDA release and compare it with CUDA 12.4."
    is_search_query = (
        "web_search" in decision.tools
        or decision.intent == "web_search"
        or "search the web" in lowered
        or "search for" in lowered
    )
    has_comparison = any(
        kw in lowered
        for kw in (
            "compare",
            "comparison",
            "newer than",
            "older than",
            "how many minor versions",
            "how many versions",
            "difference between",
            "difference with",
        )
    )

    has_fetch_keyword = any(
        kw in lowered
        for kw in (
            "read the page",
            "read the documentation",
            "fetch the page",
            "read the top result",
            "read it",
            "fetch it",
            "summarize the page",
            "summarize it",
            "summarize",
        )
    )

    # 1a. Detect Web Search + Web Fetch + Calculate 3-step chain
    # e.g. "Search for the latest Python release, read the documentation page, and compare it with 3.13."
    if is_search_query and has_fetch_keyword and (has_comparison or "calculate" in lowered or "changes" in lowered):
        entity, target_version = extract_entity_and_target_version(user_message)
        search_args = resolve_web_search_arguments(user_message, decision.tool_arguments)
        step_1 = ToolStep(
            step_id="step_1",
            tool_name="web_search",
            arguments=search_args,
            depends_on=[],
            purpose=f"Search the web for {entity} documentation or release notes",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="web_fetch",
            arguments={},
            depends_on=["step_1"],
            purpose=f"Fetch the top result page for {entity}",
            extractor="extract_search_url",
        )
        calc_extractor = "extract_version_comparison" if has_comparison else "extract_page_calculation"
        step_3 = ToolStep(
            step_id="step_3",
            tool_name="calculator",
            arguments={"target_version": target_version, "entity": entity, "user_message": user_message},
            depends_on=["step_2"],
            purpose=f"Calculate {entity} version difference or page values",
            extractor=calc_extractor,
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2, step_3])

    # 1b. Detect Web Search + Web Fetch 2-step chain
    # e.g. "Search for the FastAPI tutorial and read it."
    url_match = re.search(r"https?://[^\s>]+", user_message)
    if is_search_query and has_fetch_keyword and not url_match:
        search_args = resolve_web_search_arguments(user_message, decision.tool_arguments)
        step_1 = ToolStep(
            step_id="step_1",
            tool_name="web_search",
            arguments=search_args,
            depends_on=[],
            purpose="Search the web for relevant documentation",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="web_fetch",
            arguments={},
            depends_on=["step_1"],
            purpose="Fetch and read top search result page",
            extractor="extract_search_url",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 1c. Detect Web Search + Compare / Calculate workflow (2 steps)
    if is_search_query and has_comparison:
        entity, target_version = extract_entity_and_target_version(user_message)

        search_args = resolve_web_search_arguments(user_message, decision.tool_arguments)

        step_1 = ToolStep(
            step_id="step_1",
            tool_name="web_search",
            arguments=search_args,
            depends_on=[],
            purpose=f"Retrieve latest {entity} release information from the web",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="calculator",
            arguments={"target_version": target_version, "entity": entity, "user_message": user_message},
            depends_on=["step_1"],
            purpose=f"Calculate {entity} version difference relative to {target_version}",
            extractor="extract_version_comparison",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 1d. Detect direct Web Fetch + Compare / Calculate workflow (2 steps)
    if url_match and has_comparison and not is_search_query:
        fetch_url = url_match.group(0).rstrip(".,;!?'\")")
        entity, target_version = extract_entity_and_target_version(user_message)
        step_1 = ToolStep(
            step_id="step_1",
            tool_name="web_fetch",
            arguments=resolve_web_fetch_arguments(user_message, {"url": fetch_url}),
            depends_on=[],
            purpose=f"Fetch page content for {entity}",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="calculator",
            arguments={"target_version": target_version, "entity": entity, "user_message": user_message},
            depends_on=["step_1"],
            purpose=f"Calculate {entity} version difference relative to {target_version}",
            extractor="extract_version_comparison",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 1e. Detect direct version comparison without search or fetch (1 step calculator)
    # e.g. "Compare Python 3.14.8 with Python 3.13" or resolved follow-up
    v_comp = re.search(
        r"compare\s+.*?([vV]?\d+(?:\.\d+)+).*?(?:with|to)\s+.*?([vV]?\d+(?:\.\d+)+)",
        user_message,
        re.IGNORECASE,
    )
    if (v_comp or (decision.tool_arguments.get("discovered_version") and decision.tool_arguments.get("target_version"))) and not is_search_query and not url_match:
        if v_comp:
            v1 = v_comp.group(1).lstrip("vV")
            v2 = v_comp.group(2).lstrip("vV")
        else:
            v1 = str(decision.tool_arguments["discovered_version"]).lstrip("vV")
            v2 = str(decision.tool_arguments["target_version"]).lstrip("vV")

        p1 = [int(x) for x in re.findall(r"\d+", v1)]
        p2 = [int(x) for x in re.findall(r"\d+", v2)]
        if len(p1) >= 2 and len(p2) >= 2:
            m1, m2 = p1[1], p2[1]
            diff_expr = f"{m1} - {m2}"
            ent = "python" if "python" in lowered else decision.tool_arguments.get("entity", "Python")
            step = ToolStep(
                step_id="step_1",
                tool_name="calculator",
                arguments={
                    "expression": diff_expr,
                    "discovered_version": v1,
                    "target_version": v2,
                    "entity": ent,
                },
                purpose=f"Calculate minor version difference between {v1} and {v2}",
            )
            return ExecutionPlan(plan_id=plan_id, steps=[step])

    # 2. Detect Web Fetch + Memory Write workflow
    # e.g. "Read https://... and remember that I'm learning FastAPI."
    # e.g. "Fetch https://... and remember I prefer dark mode."
    has_remember = bool(re.search(r"\bremember(?:\s+that)?\s+(.+)$", user_message, re.IGNORECASE))

    if url_match and has_remember:
        fetch_url = url_match.group(0).rstrip(".,;!?'\")")
        fetch_args = resolve_web_fetch_arguments(user_message, {"url": fetch_url})

        rem_match = re.search(r"\bremember(?:\s+that)?\s+(.+)$", user_message, re.IGNORECASE)
        memory_content = rem_match.group(1).rstrip(".!?").strip() if rem_match else ""

        step_1 = ToolStep(
            step_id="step_1",
            tool_name="web_fetch",
            arguments=fetch_args,
            depends_on=[],
            purpose="Retrieve and analyze web page content",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="remember_memory",
            arguments={
                "content": memory_content,
                "memory_type": "fact",
                "importance": 2,
                "confirmed": bool(decision.tool_arguments.get("confirmed")),
            },
            depends_on=["step_1"],
            purpose=f"Record user memory: '{memory_content}'",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 3. Detect Datetime + Date Offset calculation
    # e.g. "What date is 10 days after today?" or "What date is 5 days from now?"
    # e.g. "What date was 10 days before today?" or "What date was 5 days ago?"
    future_match = re.search(r"\b(\d+)\s+days?\s+(?:after|from|past)\s+(?:today|now)\b", lowered)
    past_match = re.search(r"\b(\d+)\s+days?\s+(?:before|prior to)\s+(?:today|now)\b", lowered) or re.search(r"\b(\d+)\s+days?\s+ago\b", lowered)
    has_date_offset = (future_match or past_match or decision.tool_arguments.get("days") is not None)

    if has_date_offset and "datetime" in decision.tools:
        if decision.tool_arguments.get("days") is not None:
            days = int(decision.tool_arguments["days"])
        elif future_match:
            days = int(future_match.group(1))
        else:
            days = -int(past_match.group(1))

        step_1 = ToolStep(
            step_id="step_1",
            tool_name="datetime",
            arguments={"query_type": "date"},
            depends_on=[],
            purpose="Get current reference date",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="calculator",
            arguments={"days": days, "offset": days},
            depends_on=["step_1"],
            purpose=f"Calculate date offset of {days} days from current date",
            extractor="extract_date_offset",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 4. Detect Datetime + Web Search workflow (2 steps)
    # e.g. "What date is today and search for headlines today"
    has_events_search = any(kw in lowered for kw in ("headlines", "news", "events", "what happened"))
    if "datetime" in decision.tools and is_search_query and has_events_search:
        step_1 = ToolStep(
            step_id="step_1",
            tool_name="datetime",
            arguments={"query_type": "date"},
            depends_on=[],
            purpose="Get current reference date",
        )
        step_2 = ToolStep(
            step_id="step_2",
            tool_name="web_search",
            arguments=resolve_web_search_arguments(user_message, decision.tool_arguments),
            depends_on=["step_1"],
            purpose="Search the web with reference date",
            extractor="extract_date_for_search",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step_1, step_2])

    # 5. Handle multiple tools already declared in decision.tools
    # e.g. ["calculator", "datetime"] from policy
    if len(decision.tools) > 1:
        steps: list[ToolStep] = []
        for idx, tool_name in enumerate(decision.tools[:MAX_PLAN_STEPS], start=1):
            step_args: dict[str, Any] = {}
            if tool_name == "calculator":
                step_args = resolve_calculator_arguments(user_message)
            elif tool_name == "datetime":
                step_args = resolve_datetime_arguments(user_message, decision.tool_arguments)
            elif tool_name == "web_search":
                step_args = resolve_web_search_arguments(user_message, decision.tool_arguments)
            elif tool_name == "web_fetch":
                step_args = resolve_web_fetch_arguments(user_message, decision.tool_arguments)
            else:
                step_args = dict(decision.tool_arguments)

            steps.append(
                ToolStep(
                    step_id=f"step_{idx}",
                    tool_name=tool_name,
                    arguments=step_args,
                    depends_on=[],
                    purpose=f"Execute {tool_name}",
                )
            )
        return ExecutionPlan(plan_id=plan_id, steps=steps)

    # 6. Handle single tool request
    if len(decision.tools) == 1:
        tool_name = decision.tools[0]
        step_args = {}
        if tool_name == "calculator":
            step_args = resolve_calculator_arguments(user_message)
        elif tool_name == "datetime":
            step_args = resolve_datetime_arguments(user_message, decision.tool_arguments)
        elif tool_name == "web_search":
            step_args = resolve_web_search_arguments(user_message, decision.tool_arguments)
        elif tool_name == "web_fetch":
            step_args = resolve_web_fetch_arguments(user_message, decision.tool_arguments)
        else:
            step_args = dict(decision.tool_arguments)

        step = ToolStep(
            step_id="step_1",
            tool_name=tool_name,
            arguments=step_args,
            depends_on=[],
            purpose=f"Execute {tool_name}",
        )
        return ExecutionPlan(plan_id=plan_id, steps=[step])

    # 7. No tools required (local or memory route)
    return ExecutionPlan(plan_id=plan_id, steps=[])


class ControlledGeneralPlanner:
    """
    Controlled General Planner (P7).
    Decomposes user requests into bounded ExecutionPlans using structured candidate schemas,
    optional LLM decomposition, deterministic plan validation, and deterministic fallbacks.
    """

    def __init__(self, llm: Any = None, tool_registry: Any = None, use_llm: bool | None = None):
        self.llm = llm
        self.tool_registry = tool_registry
        self.last_trace: Any = None
        if use_llm is not None:
            self.use_llm = use_llm
        else:
            from app.core.config import settings
            self.use_llm = getattr(settings, "enable_llm_planner", False)

    def plan(
        self,
        user_message: str,
        decision: AgentDecision | None = None,
        user_id: str | None = None,
        active_task: Any = None,
    ) -> tuple[ExecutionPlan, Any]:
        """
        Produce a validated ExecutionPlan and an ExecutionTrace.
        Pipeline: Capability Decomposition -> Candidate Scoring -> Plan Selection -> Deterministic Validation -> Fallback if invalid.
        """
        from app.agent.trace import ExecutionTrace
        from app.agent.planner_schema import CandidatePlan
        from app.agent.policy import apply_policy
        from app.agent.capabilities import (
            decompose_request_capabilities,
            score_tool_candidates,
            BUILTIN_CAPABILITIES,
        )

        if decision is None:
            initial_decision = AgentDecision(
                intent="multi_tool",
                route="tool",
                needs_clarification=False,
                tools=[],
                tool_arguments={},
                reason="General planner",
            )
            decision = apply_policy(user_message, initial_decision)

        trace = ExecutionTrace(
            plan_id="",
            user_message=user_message,
        )
        self.last_trace = trace

        # P10: Decompose request into explicit capabilities and score candidates
        required_caps = decompose_request_capabilities(user_message, active_task=active_task)
        scored_candidates = score_tool_candidates(required_caps, user_message, active_task=active_task)

        trace.requested_capabilities = [c.value for c in required_caps]
        trace.candidate_tools = [cand[0] for cand in scored_candidates]

        from app.agent.executor import PlanExecutor
        executor = PlanExecutor(self.tool_registry) if self.tool_registry else None

        # 1. No tool or local route -> empty plan
        if (
            decision.route in ("local", "memory")
            or (decision.route == "tool" and not decision.tools and decision.intent != "multi_tool" and not required_caps)
        ):
            empty_plan = ExecutionPlan(plan_id=f"plan_{uuid.uuid4().hex[:8]}", steps=[])
            trace.plan_id = empty_plan.plan_id
            trace.planner_source = "deterministic"
            trace.validation_passed = True
            trace.final_status = "success"
            trace.final_validated_plan = empty_plan.to_dict()
            return empty_plan, trace

        # 2. Attempt LLM decomposition if LLM is available and suitable
        attempted_llm = False
        if (
            self.use_llm
            and self.llm is not None
            and hasattr(self.llm, "client")
            and getattr(self.llm, "client", None) is not None
            and (decision.intent == "multi_tool" or len(decision.tools) > 1 or " and " in user_message.lower())
        ):
            attempted_llm = True
            try:
                candidate = self._decompose_with_llm(user_message, decision)
                if candidate is not None:
                    # Check for unknown tools in LLM plan
                    unknown_tools = [s.tool_name for s in candidate.steps if s.tool_name not in BUILTIN_CAPABILITIES]
                    if unknown_tools:
                        trace.validation_error = f"LLM proposed unknown tool(s): {', '.join(unknown_tools)}"
                        for ut in unknown_tools:
                            trace.rejected_candidate_reasons[ut] = f"Tool '{ut}' is not a registered capability"
                    else:
                        plan_steps = [
                            ToolStep(
                                step_id=s.step_id,
                                tool_name=s.tool_name,
                                arguments=s.arguments,
                                depends_on=s.depends_on,
                                purpose=s.purpose,
                                extractor=s.extractor,
                            )
                            for s in candidate.steps
                        ]
                        candidate_exec_plan = ExecutionPlan(plan_id=candidate.plan_id, steps=plan_steps)
                        if executor:
                            is_valid, err = executor.validate_plan(candidate_exec_plan, user_message=user_message)
                            if is_valid:
                                trace.plan_id = candidate_exec_plan.plan_id
                                trace.planner_source = "llm"
                                trace.candidate_plan = candidate.model_dump()
                                trace.validation_passed = True
                                trace.selected_tools = [s.tool_name for s in candidate_exec_plan.steps]
                                for cand_name, score, reasons in scored_candidates:
                                    r_str = "; ".join(f"{k}: {v}" for k, v in reasons.items()) if reasons else f"Score {score}"
                                    if cand_name in trace.selected_tools:
                                        trace.selection_reasons[cand_name] = r_str
                                    else:
                                        trace.rejected_candidate_reasons[cand_name] = r_str
                                trace.final_validated_plan = candidate_exec_plan.to_dict()
                                return candidate_exec_plan, trace
                            else:
                                trace.validation_error = f"LLM plan validation failed: {err}"
                                trace.candidate_plan = candidate.model_dump()
            except Exception as e:
                trace.validation_error = f"LLM decomposition failed: {e}"

        # 3. Deterministic Planner Fallback
        det_plan = build_execution_plan(user_message, decision, user_id, active_task=active_task)
        trace.plan_id = det_plan.plan_id
        trace.planner_source = "fallback" if attempted_llm else "deterministic"
        trace.selected_tools = [s.tool_name for s in det_plan.steps]

        for cand_name, score, reasons in scored_candidates:
            r_str = "; ".join(f"{k}: {v}" for k, v in reasons.items()) if reasons else f"Score {score}"
            if cand_name in trace.selected_tools:
                trace.selection_reasons[cand_name] = r_str
            else:
                trace.rejected_candidate_reasons[cand_name] = r_str

        if executor:
            is_valid, err = executor.validate_plan(det_plan, user_message=user_message)
            trace.validation_passed = is_valid
            if not is_valid:
                trace.validation_error = err
        else:
            trace.validation_passed = True

        if trace.validation_passed:
            trace.final_validated_plan = det_plan.to_dict()

        return det_plan, trace

    def _decompose_with_llm(self, user_message: str, decision: AgentDecision) -> Any:
        """Prompt LLM for structured candidate plan."""
        from app.agent.capabilities import BUILTIN_CAPABILITIES
        from app.agent.plan import KNOWN_EXTRACTORS
        from app.agent.planner_schema import CandidatePlan

        extractors_desc = ", ".join(KNOWN_EXTRACTORS)
        allowed_tools_str = ", ".join(BUILTIN_CAPABILITIES.keys())

        system_prompt = (
            "You are the MAI Execution Planner. Decompose the user request into a bounded list of tool steps.\n"
            "RULES:\n"
            "1. Maximum 4 steps.\n"
            "2. Step IDs must be unique ('step_1', 'step_2', etc.).\n"
            "3. Dependencies must reference previous step IDs.\n"
            f"4. Only use allowed tools: {allowed_tools_str}.\n"
            f"5. If a step consumes output from a previous step, specify an allowed extractor: {extractors_desc}.\n"
            "6. Never create memory mutation steps ('remember_memory') unless user explicitly requested to remember.\n"
            "7. Return JSON matching the schema."
        )

        response = self.llm.client.chat(
            model=getattr(self.llm, "model", "qwen2.5:3b"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Decompose this request: {user_message}"},
            ],
            format=CandidatePlan.model_json_schema(),
        )
        content = response["message"]["content"]
        return CandidatePlan.model_validate_json(content)
