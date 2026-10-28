"""
P16.1 — Nemotron Procedural Memory Assistant.
Coordinates with OpenRouter NVIDIA Nemotron for structured skill proposals,
adaptation to new tasks, and reusable workflow summarization.
All model output is treated as untrusted data with schema validation and security filtering.
"""
import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMProvider
from app.skills.model import (
    AdaptedPlanModel,
    SkillProposal,
    SkillProposalModel,
    SkillSecurityViolationError,
    SkillValidationError,
    WorkflowSummaryModel,
    validate_skill_security,
)
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_PROPOSAL_STEPS = 10

# Regex patterns for sanitization before external LLM calls
SENSITIVE_PATTERNS = [
    re.compile(r"(?:api[_-]?key|bearer\s+[a-zA-Z0-9_\-\.]+|password|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"\bAIzaSy[a-zA-Z0-9_\-]{20,}\b"),
    re.compile(r"\bgsk_[a-zA-Z0-9_\-]{20,}\b"),
    re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{20,}|ghp_[a-zA-Z0-9_\-]{20,})\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
]


def sanitize_trace_for_llm(trace_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize an execution trace before sending it to an external LLM.
    PRIVACY GUARANTEE: Removes credentials, API keys, emails, and truncates large payloads.
    """
    sanitized: dict[str, Any] = {}

    for k, v in trace_dict.items():
        if k in ("api_key", "password", "token", "secret", "user_credentials", "auth"):
            continue

        if isinstance(v, str):
            clean_str = v
            for pat in SENSITIVE_PATTERNS:
                clean_str = pat.sub("<REDACTED>", clean_str)
            # Truncate verbose text payloads to 500 characters
            if len(clean_str) > 500:
                clean_str = clean_str[:500] + "... [truncated]"
            sanitized[k] = clean_str

        elif isinstance(v, dict):
            sanitized[k] = sanitize_trace_for_llm(v)

        elif isinstance(v, list):
            clean_list = []
            for item in v:
                if isinstance(item, dict):
                    clean_list.append(sanitize_trace_for_llm(item))
                elif isinstance(item, str):
                    c_str = item
                    for pat in SENSITIVE_PATTERNS:
                        c_str = pat.sub("<REDACTED>", c_str)
                    if len(c_str) > 500:
                        c_str = c_str[:500] + "... [truncated]"
                    clean_list.append(c_str)
                else:
                    clean_list.append(item)
            sanitized[k] = clean_list

        else:
            sanitized[k] = v

    return sanitized


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from raw LLM output."""
    if not text or not text.strip():
        raise ValueError("Empty response received from LLM.")

    # 1. Try markdown code block extraction
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    # 2. Try outermost braces
    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(1))

    # 3. Direct JSON parse
    return json.loads(text.strip())


class NemotronSkillAssistant:
    """
    Assistant coordinating with NVIDIA Nemotron for procedural memory tasks.
    Enforces schema validation, single bounded repair attempt, and tool allowlisting.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry

    def _get_allowed_tool_names(self) -> set[str]:
        """Return allowed tool names from registry or default builtin tools."""
        if self.tool_registry:
            tools = self.tool_registry.list_tools()
            if isinstance(tools, dict):
                names = set(tools.keys())
            else:
                names = {t.name for t in tools}
            # Include memory and builtin utilities
            names.update({"remember_memory", "search_memory", "update_memory", "forget_memory"})
            return names
        return {
            "web_search",
            "web_fetch",
            "calculator",
            "datetime",
            "remember_memory",
            "search_memory",
            "update_memory",
            "forget_memory",
        }

    def propose_skill_from_trace(
        self,
        trace_data: dict[str, Any],
        user_goal: str | None = None,
    ) -> SkillProposal:
        """
        Convert a successful task execution trace into a structured, validated SkillProposal.
        Uses Nemotron with single bounded repair if JSON or schema validation fails.
        """
        sanitized_trace = sanitize_trace_for_llm(trace_data)
        allowed_tools = self._get_allowed_tool_names()

        system_prompt = (
            "You are P-MAI's procedural memory architect. Your job is to convert successful task "
            "traces into structured, reusable procedural skills.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a single valid JSON object adhering strictly to the schema.\n"
            "2. Do NOT generate arbitrary code, shell commands, or bash scripts.\n"
            "3. Only use tools from this allowed list: " + ", ".join(sorted(allowed_tools)) + ".\n"
            "4. Maximum steps allowed: 10.\n"
            "5. Identify reusable input parameters and represent them in arguments as {param_name}.\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "name": "Short Descriptive Title",\n'
            '  "description": "Clear explanation of what the skill does",\n'
            '  "intended_use": "When to apply this skill",\n'
            '  "parameters_schema": {"param_name": {"type": "string", "required": true, "description": "..."}},\n'
            '  "steps": [\n'
            '    {"step_id": "step_1", "tool_name": "...", "arguments": {...}, "depends_on": [], "purpose": "..."}\n'
            "  ],\n"
            '  "expected_outputs": "...",\n'
            '  "verification_criteria": "...",\n'
            '  "known_limitations": "..."\n'
            "}"
        )

        user_content = (
            f"User Goal: {user_goal or sanitized_trace.get('user_goal', 'Multi-step task')}\n"
            f"Execution Trace:\n{json.dumps(sanitized_trace, indent=2)}\n\n"
            "Propose a reusable procedural skill matching this workflow."
        )

        messages = [{"role": "user", "content": user_content}]

        # 1. Initial attempt
        raw_output = self.llm.generate(messages=messages, system_prompt=system_prompt)
        proposal_model, err = self._validate_proposal_output(raw_output, allowed_tools)

        # 2. Single bounded repair attempt if needed
        if err is not None:
            logger.info("Nemotron skill proposal failed initial validation: %s. Initiating repair attempt.", err)
            repair_user = (
                f"Your previous JSON output was invalid for the following reason:\n{err}\n\n"
                "Please fix the error and return ONLY the corrected, valid JSON object adhering strictly to the schema."
            )
            repair_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": repair_user},
            ]
            repaired_output = self.llm.generate(messages=repair_messages, system_prompt=system_prompt)
            proposal_model, err2 = self._validate_proposal_output(repaired_output, allowed_tools)
            if err2 is not None:
                raise SkillValidationError(f"Nemotron failed to produce a valid skill proposal after repair attempt: {err2}")

        assert proposal_model is not None

        # Convert Pydantic model to domain SkillProposal
        steps_dicts = [s.model_dump() for s in proposal_model.steps]
        exp_out = proposal_model.expected_outputs
        if isinstance(exp_out, list):
            exp_out = "; ".join(exp_out)
        verif_crit = proposal_model.verification_criteria
        if isinstance(verif_crit, list):
            verif_crit = "; ".join(verif_crit)
        known_lim = proposal_model.known_limitations
        if isinstance(known_lim, list):
            known_lim = "; ".join(known_lim)

        return SkillProposal(
            name=proposal_model.name.strip(),
            description=proposal_model.description.strip(),
            intended_use=proposal_model.intended_use.strip(),
            parameters_schema=proposal_model.parameters_schema,
            steps=steps_dicts,
            expected_outputs=exp_out,
            verification_criteria=verif_crit,
            known_limitations=known_lim,
            provenance="nemotron_derived_from_trace",
            derived_from_task_id=str(sanitized_trace.get("task_id", "")),
        )

    def _validate_proposal_output(
        self,
        raw_text: str,
        allowed_tools: set[str],
    ) -> tuple[SkillProposalModel | None, str | None]:
        """Validate raw text output against SkillProposalModel and security constraints."""
        try:
            data = extract_json_object(raw_text)
        except Exception as e:
            return None, f"JSON parse error: {e}"

        try:
            model = SkillProposalModel.model_validate(data)
        except ValidationError as ve:
            return None, f"Schema validation error: {ve}"

        # Check step count
        if len(model.steps) > MAX_PROPOSAL_STEPS:
            return None, f"Step count ({len(model.steps)}) exceeds maximum limit of {MAX_PROPOSAL_STEPS}."

        if not model.steps:
            return None, "Skill proposal must contain at least 1 step."

        # Check tool names against allowed tools
        for step in model.steps:
            if step.tool_name not in allowed_tools:
                return None, f"Tool '{step.tool_name}' in step '{step.step_id}' is not in the allowed tool registry."

        # Security check: no eval, rm -rf, credentials, etc.
        try:
            steps_dicts = [s.model_dump() for s in model.steps]
            validate_skill_security(model.name, model.description, steps_dicts, model.parameters_schema)
        except SkillSecurityViolationError as se:
            return None, f"Security violation: {se}"

        return model, None

    def adapt_skill_for_goal(
        self,
        skill_dict: dict[str, Any],
        new_user_goal: str,
    ) -> AdaptedPlanModel:
        """
        Adapt an approved saved skill to a new user request by substituting input arguments.
        Preserves original workflow purpose and structure.
        """
        allowed_tools = self._get_allowed_tool_names()
        sanitized_skill = sanitize_trace_for_llm(skill_dict)

        system_prompt = (
            "You are P-MAI's procedural plan adapter. Adapt the given approved skill to the new user goal.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY a valid JSON object adhering to the schema.\n"
            "2. Preserve the exact sequence and purpose of the skill steps.\n"
            "3. Substitute concrete values into step arguments based on the new user goal.\n"
            "4. Only use tools from this allowed list: " + ", ".join(sorted(allowed_tools)) + ".\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "user_goal": "...",\n'
            '  "steps": [\n'
            '    {"step_id": "step_1", "tool_name": "...", "arguments": {...}, "depends_on": [], "purpose": "..."}\n'
            "  ],\n"
            '  "explanation": "Brief explanation of substitutions"\n'
            "}"
        )

        user_content = (
            f"Approved Skill:\n{json.dumps(sanitized_skill, indent=2)}\n\n"
            f"New User Goal: {new_user_goal}\n\n"
            "Adapt this skill for the new user goal."
        )

        messages = [{"role": "user", "content": user_content}]

        # Initial attempt
        raw_output = self.llm.generate(messages=messages, system_prompt=system_prompt)
        plan_model, err = self._validate_adapted_plan(raw_output, allowed_tools)

        # Single bounded repair if needed
        if err is not None:
            logger.info("Nemotron plan adaptation failed initial validation: %s. Initiating repair attempt.", err)
            repair_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": f"Your JSON output was invalid:\n{err}\nReturn ONLY valid JSON."},
            ]
            repaired_output = self.llm.generate(messages=repair_messages, system_prompt=system_prompt)
            plan_model, err2 = self._validate_adapted_plan(repaired_output, allowed_tools)
            if err2 is not None:
                raise SkillValidationError(f"Nemotron failed to adapt skill after repair: {err2}")

        assert plan_model is not None
        return plan_model

    def _validate_adapted_plan(
        self,
        raw_text: str,
        allowed_tools: set[str],
    ) -> tuple[AdaptedPlanModel | None, str | None]:
        """Validate raw adapted plan output."""
        try:
            data = extract_json_object(raw_text)
        except Exception as e:
            return None, f"JSON parse error: {e}"

        try:
            model = AdaptedPlanModel.model_validate(data)
        except ValidationError as ve:
            return None, f"Schema validation error: {ve}"

        if len(model.steps) > MAX_PROPOSAL_STEPS:
            return None, f"Step count ({len(model.steps)}) exceeds maximum limit of {MAX_PROPOSAL_STEPS}."

        for s in model.steps:
            if s.tool_name not in allowed_tools:
                return None, f"Tool '{s.tool_name}' is not in the allowed tool registry."

        return model, None

    def summarize_workflow_outcome(
        self,
        skill_name: str,
        task_goal: str,
        step_results: list[dict[str, Any]],
    ) -> WorkflowSummaryModel:
        """
        Summarize a successful execution into non-sensitive, reusable procedural metadata.
        """
        sanitized_results = [sanitize_trace_for_llm(r) for r in step_results]

        system_prompt = (
            "You are P-MAI's workflow summarizer. Summarize this successful skill run into "
            "reusable, non-sensitive metadata for future runs.\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "summary": "Concise overview of what was accomplished",\n'
            '  "reusable_insights": ["Insight 1", "Insight 2"],\n'
            '  "recommended_parameters": ["param1"]\n'
            "}"
        )

        user_content = (
            f"Skill: {skill_name}\n"
            f"Goal: {task_goal}\n"
            f"Step Results:\n{json.dumps(sanitized_results, indent=2)}\n\n"
            "Summarize the procedural outcome."
        )

        raw_output = self.llm.generate(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=system_prompt,
        )

        try:
            data = extract_json_object(raw_output)
            return WorkflowSummaryModel.model_validate(data)
        except Exception:
            # Fallback to deterministic summary if LLM output fails
            return WorkflowSummaryModel(
                summary=f"Successfully executed skill '{skill_name}' across {len(step_results)} steps.",
                reusable_insights=["Completed deterministically"],
                recommended_parameters=[],
            )
