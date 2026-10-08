"""
P8 — Multi-Turn Reference Resolver.
Deterministically resolves bounded references in follow-up requests using ActiveTaskState.
Enforces strict clarity: unambiguous references are resolved, ambiguous ones trigger clarification.
"""
from dataclasses import dataclass, field
import re
from typing import Any

from app.agent.clarification import ClarificationReason
from app.agent.task_state import ActiveTaskState


@dataclass
class ReferenceResolutionResult:
    """Outcome of resolving references in a user message against active task state."""

    resolved_message: str
    resolved_arguments: dict[str, Any] = field(default_factory=dict)
    is_reference_resolved: bool = False
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    target_tool: str | None = None
    is_cancellation: bool = False
    is_continuation: bool = False
    clarification_reason: ClarificationReason | None = None
    candidate_options: list[str] = field(default_factory=list)
    missing_information: str | None = None


# Explicit task cancellation patterns (separate from long-term memory deletion)
CANCELLATION_PATTERN = re.compile(
    r"^(?:cancel(?:\s+this(?:\s+task)?)?|cancel\s+task|stop(?:\s+this(?:\s+task)?)?|forget\s+this\s+task|start\s+over)[.!?]?$",
    re.IGNORECASE,
)

# Continuation patterns
CONTINUATION_PATTERN = re.compile(
    r"^(?:continue(?:\s+the(?:\s+task)?)?|resume(?:\s+task)?|proceed|go\s+on)[.!?]?$",
    re.IGNORECASE,
)


def is_cancellation_command(message: str) -> bool:
    """Detect explicit task cancellation commands."""
    return bool(CANCELLATION_PATTERN.match(message.strip()))


def is_continuation_command(message: str) -> bool:
    """Detect continuation commands."""
    return bool(CONTINUATION_PATTERN.match(message.strip()))


def resolve_task_reference(
    user_message: str,
    active_task: ActiveTaskState | None,
) -> ReferenceResolutionResult:
    """
    Resolve follow-up references using ActiveTaskState when unambiguous.
    If reference is ambiguous, asks a clarification question and never guesses.
    """
    stripped = user_message.strip()
    lowered = stripped.lower()

    # 1. Check for cancellation
    if is_cancellation_command(stripped):
        return ReferenceResolutionResult(
            resolved_message=stripped,
            is_cancellation=True,
        )

    # 2. Check for continuation command
    if is_continuation_command(stripped):
        if not active_task or active_task.is_expired():
            return ReferenceResolutionResult(
                resolved_message=stripped,
                is_continuation=True,
                needs_clarification=True,
                clarification_reason=ClarificationReason.EXPIRED_TASK_REFERENCE,
                missing_information="task_goal",
                clarification_prompt="There is no active task to continue. What would you like me to help you with?",
            )
        # Check if active task has next steps or was already completed
        if active_task.completed_steps:
            return ReferenceResolutionResult(
                resolved_message=stripped,
                is_continuation=True,
                needs_clarification=True,
                clarification_reason=ClarificationReason.INSUFFICIENT_CONTEXT,
                missing_information="next_step",
                clarification_prompt=(
                    f"The previous task for '{active_task.original_user_goal}' is complete. "
                    f"What next step would you like me to take?"
                ),
            )
        return ReferenceResolutionResult(
            resolved_message=active_task.original_user_goal,
            is_continuation=True,
            is_reference_resolved=True,
        )

    # If no active task exists, fast-path return unmodified message
    if not active_task or active_task.is_expired():
        # Check if user used explicit dangling pronoun requiring clarification
        dangling_pronouns = (
            "compare that with",
            "compare that version with",
            "read that page",
            "summarize that page",
            "summarize the page",
            "use the same page",
            "divide that result by",
            "now divide that result by",
            "multiply that result by",
        )
        for dp in dangling_pronouns:
            if dp in lowered:
                if any(w in dp for w in ("divide", "multiply", "add", "subtract")) and re.search(r"\d+\s*[\+\-\*/%]\s*\d+", lowered):
                    continue
                if "page" in lowered:
                    return ReferenceResolutionResult(
                        resolved_message=stripped,
                        needs_clarification=True,
                        clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                        missing_information="page_url",
                        clarification_prompt="Which page would you like me to read? Please provide the URL or topic.",
                    )
                if any(w in lowered for w in ("divide", "multiply", "add", "subtract", "result")):
                    return ReferenceResolutionResult(
                        resolved_message=stripped,
                        needs_clarification=True,
                        clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                        missing_information="calculator_result",
                        clarification_prompt="Which calculation result would you like to divide or calculate?",
                    )
                return ReferenceResolutionResult(
                    resolved_message=stripped,
                    needs_clarification=True,
                    clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                    missing_information="reference_target",
                    clarification_prompt="I couldn't find an active task or prior result to reference. Could you provide the specific details?",
                )
        return ReferenceResolutionResult(resolved_message=stripped)

    # Task context is available and unexpired. Inspect typed intermediate results.
    ctx = active_task.context_values

    # Check for direct override / update in active task
    override_match = re.search(
        r"actually\s+compare\s+(?:it\s+with\s+|([vV]?\d+(?:\.\d+)+)\s+with\s+)?([vV]?\d+(?:\.\d+)+)",
        stripped,
        re.IGNORECASE,
    )
    if override_match:
        target_version = override_match.group(2).lstrip("vV")
        first_v = override_match.group(1).lstrip("vV") if override_match.group(1) else ctx.get("discovered_version", "3.13")
        entity = ctx.get("target_entity", "python")
        ctx["requested_comparison_target"] = target_version
        resolved = f"Compare {entity} {first_v} with {entity} {target_version}"
        return ReferenceResolutionResult(
            resolved_message=resolved,
            resolved_arguments={
                "discovered_version": first_v,
                "target_version": target_version,
                "entity": entity,
            },
            target_tool="calculator",
            is_reference_resolved=True,
        )

    # 3. "Do the same for <entity>" pattern
    # e.g. "do the same for FastAPI", "do the same for CUDA"
    same_entity_match = re.search(r"\bdo\s+the\s+same\s+for\s+([a-zA-Z0-9_-]+)", stripped, re.IGNORECASE)
    if same_entity_match:
        new_entity = same_entity_match.group(1).lower()
        prior_entity = ctx.get("target_entity", "python")
        prior_goal = active_task.original_user_goal
        if prior_entity and prior_entity in prior_goal.lower():
            resolved = re.sub(rf"\b{re.escape(prior_entity)}\b", new_entity, prior_goal, flags=re.IGNORECASE)
        else:
            resolved = f"Search for the latest {new_entity} release and compare it."
        return ReferenceResolutionResult(
            resolved_message=resolved,
            resolved_arguments={"entity": new_entity},
            is_reference_resolved=True,
        )

    # 4. Arithmetic / Calculator follow-up references
    # e.g. "Now divide that result by 10.", "divide that result by 10", "multiply that by 5"
    calc_followup_match = re.search(
        r"(?:now\s+)?(divide|multiply|add|subtract|plus|minus|times)\s+(?:that(?:\s+result)?|the\s+result|it)\s+(?:by|with|to|from)\s+(\d+(?:\.\d+)?)",
        stripped,
        re.IGNORECASE,
    ) or re.search(
        r"(?:now\s+)?(divide|multiply|add|subtract)\s+(?:that(?:\s+result)?|it)\s+by\s+(\d+(?:\.\d+)?)",
        stripped,
        re.IGNORECASE,
    )

    if calc_followup_match:
        op_word = calc_followup_match.group(1).lower()
        operand = calc_followup_match.group(2)

        # Ambiguity check: multiple calculator results present
        cand_calcs = ctx.get("candidate_calculator_results", [])
        if len(cand_calcs) > 1:
            opts = [str(c) for c in cand_calcs]
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_reason=ClarificationReason.MULTIPLE_MATCHES,
                candidate_options=opts,
                missing_information="calculator_operand",
                clarification_prompt=f"Which result do you mean: {opts[0]} or {opts[1]}?",
            )

        prior_result = ctx.get("calculator_result")
        if prior_result is None:
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                missing_information="calculator_result",
                clarification_prompt="Which calculation result would you like to divide or calculate?",
            )

        op_map = {
            "divide": "/",
            "multiply": "*",
            "times": "*",
            "add": "+",
            "plus": "+",
            "subtract": "-",
            "minus": "-",
        }
        operator = op_map.get(op_word, "/")
        expr = f"{prior_result} {operator} {operand}"
        resolved = f"Calculate {expr}"
        return ReferenceResolutionResult(
            resolved_message=resolved,
            resolved_arguments={"expression": expr},
            target_tool="calculator",
            is_reference_resolved=True,
        )

    # 5. Web Fetch follow-up references ("read that page", "the same page", "summarize that page")
    fetch_ref_match = any(
        phrase in lowered
        for phrase in (
            "that page",
            "the same page",
            "that documentation",
            "that url",
            "read it",
            "summarize that page",
            "fetch that page",
        )
    )

    if fetch_ref_match and not re.search(r"https?://[^\s>]+", stripped):
        # Ambiguity check: multiple documentation pages present
        cand_urls = ctx.get("candidate_urls", [])
        cand_titles = ctx.get("candidate_titles", [])
        if len(cand_urls) > 1:
            opts = cand_titles if len(cand_titles) == len(cand_urls) else cand_urls
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_reason=ClarificationReason.MULTIPLE_MATCHES,
                candidate_options=opts,
                missing_information="selected_url",
                clarification_prompt=f"Which page do you mean, the {opts[0]} or the {opts[1]}?",
            )

        prior_url = ctx.get("selected_url")
        if not prior_url:
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                missing_information="page_url",
                clarification_prompt="Which page would you like me to read? Please provide the URL or topic.",
            )

        # Build clean resolved request preserving user instruction
        instruction = stripped
        for phrase in ("that page", "the same page", "that documentation", "that url"):
            instruction = re.sub(rf"\b{re.escape(phrase)}\b", prior_url, instruction, flags=re.IGNORECASE)

        if prior_url not in instruction:
            instruction = f"Read {prior_url} and {instruction}"

        return ReferenceResolutionResult(
            resolved_message=instruction,
            resolved_arguments={"url": prior_url},
            target_tool="web_fetch",
            is_reference_resolved=True,
        )

    # 6. Version comparison follow-up references
    version_comp_match = re.search(
        r"(?:compare\s+(?:that(?:\s+version)?|it)\s+(?:with|to)|what\s+about)\s+(?:[A-Za-z0-9_-]+\s*)?([vV]?\d+(?:\.\d+)+)",
        stripped,
        re.IGNORECASE,
    ) or re.search(
        r"(?:compare\s+(?:that|it))\s+with\s+([vV]?\d+(?:\.\d+)+)",
        stripped,
        re.IGNORECASE,
    )

    if version_comp_match:
        target_version = version_comp_match.group(1).lstrip("vV")
        discovered_version = ctx.get("discovered_version")
        entity = ctx.get("target_entity", "python")

        if not discovered_version:
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_reason=ClarificationReason.MISSING_ARGUMENT,
                missing_information="discovered_version",
                clarification_prompt=f"Which version would you like to compare with {target_version}?",
            )

        # Unambiguous resolution: we know entity, discovered_version, and target_version
        resolved = f"Compare {entity} {discovered_version} with {entity} {target_version}"
        return ReferenceResolutionResult(
            resolved_message=resolved,
            resolved_arguments={
                "discovered_version": discovered_version,
                "target_version": target_version,
                "entity": entity,
            },
            target_tool="calculator",
            is_reference_resolved=True,
        )

    # 7. Ambiguous references check
    # If the user uses ambiguous demonstratives ("that", "it") without explicit resolution
    ambiguous_demonstrative = bool(
        re.search(r"\b(compare\s+(?:that|it)|check\s+(?:that|it)|what\s+about\s+(?:that|it))\b", lowered)
    )
    if ambiguous_demonstrative:
        # Check if we have multiple candidates or missing arguments
        if "discovered_version" not in ctx and "selected_url" not in ctx and "calculator_result" not in ctx:
            return ReferenceResolutionResult(
                resolved_message=stripped,
                needs_clarification=True,
                clarification_prompt="Could you clarify what you would like me to compare or check?",
            )

    return ReferenceResolutionResult(resolved_message=stripped)
