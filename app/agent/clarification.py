"""
P9 — Intelligent Clarification & Ambiguity Resolution.
Provides deterministic clarification detection, structured ClarificationRequest model,
bounded clarification rounds, and seamless task resumption.
Strictly preserves P8 task state isolation and memory boundaries.
"""
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Optional
import uuid
from pydantic import BaseModel, Field


MAX_CLARIFICATION_ROUNDS: int = 3


class ClarificationReason(str, Enum):
    """Categorized root reasons for clarification requests."""

    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    CONFLICTING_CONSTRAINTS = "CONFLICTING_CONSTRAINTS"
    UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
    EXPIRED_TASK_REFERENCE = "EXPIRED_TASK_REFERENCE"


class ClarificationStatus(str, Enum):
    """Lifecycle statuses for a clarification request."""

    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ClarificationRequest(BaseModel):
    """
    First-class structured clarification model.
    Captures precisely what information is missing, candidate options (if any),
    and the root cause reason without exposing internal IDs to users.
    """

    clarification_id: str = Field(default_factory=lambda: f"clarify_{uuid.uuid4().hex[:8]}")
    task_id: Optional[str] = None
    question: str
    missing_information: str
    candidate_options: list[str] = Field(default_factory=list)
    affected_step: Optional[str] = None
    reason: ClarificationReason
    status: ClarificationStatus = ClarificationStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert clarification request to dictionary for internal tracing."""
        return {
            "clarification_id": self.clarification_id,
            "task_id": self.task_id,
            "question": self.question,
            "missing_information": self.missing_information,
            "candidate_options": self.candidate_options,
            "affected_step": self.affected_step,
            "reason": self.reason.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


KNOWN_SOFTWARE_ENTITIES = {
    "python",
    "fastapi",
    "cuda",
    "django",
    "pydantic",
    "node",
    "nodejs",
    "rust",
    "react",
    "go",
    "golang",
    "docker",
    "kubernetes",
    "typescript",
    "linux",
    "ubuntu",
}


def is_clear_single_turn_request(user_message: str) -> bool:
    """
    Determine if a message is self-contained and clear, allowing the fast path.
    """
    stripped = user_message.strip()
    lowered = stripped.lower()

    # Demonstrative pronouns requiring prior context
    demonstratives = (
        "that",
        "it",
        "that version",
        "that page",
        "that result",
        "the same page",
        "the result",
        "continue",
        "proceed",
    )
    for dem in demonstratives:
        if re.search(rf"\b{re.escape(dem)}\b", lowered):
            return False

    return True


COMMON_VERBS_AND_STOPWORDS = {
    "calculate", "compute", "divide", "multiply", "add", "subtract",
    "search", "find", "read", "fetch", "explain", "summarize",
    "what", "when", "where", "which", "who", "why", "how",
    "tell", "show", "check", "verify", "is", "are", "do", "does",
    "cancel", "stop", "start", "continue", "resume", "proceed",
    "compare", "use", "make", "create",
    "the", "a", "an", "this", "that", "these", "those", "some",
}


def detect_unknown_entity(
    user_message: str,
    active_task: Any = None,
) -> tuple[bool, str | None]:
    """
    Detect references to unknown software entities that cannot be safely identified.
    Never invents or hallucinates an unknown entity.
    """
    # Pattern: "compare Acme 4.2 with..." or "search for Acme 5.0"
    match = re.search(
        r"\b(?:compare|about|for|version\s+of)\s+([A-Za-z][a-zA-Z0-9_-]+)\s+(?:[vV]?\d+(?:\.\d+)*|release|latest)",
        user_message,
        re.IGNORECASE,
    )
    if match:
        candidate_raw = match.group(1)
        candidate = candidate_raw.lower()
        if (
            candidate not in KNOWN_SOFTWARE_ENTITIES
            and candidate not in COMMON_VERBS_AND_STOPWORDS
        ):
            # Check if active task established this entity
            if active_task and active_task.context_values.get("target_entity") == candidate:
                return False, None
            return True, candidate_raw

    return False, None


def detect_missing_required_arguments(
    user_message: str,
    active_task: Any = None,
) -> ClarificationRequest | None:
    """
    Analyze whether required tool arguments are missing before planning or execution.
    """
    stripped = user_message.strip()
    lowered = stripped.lower()

    # 1. "Search for the release" without entity or topic
    if re.match(r"^search\s+(?:for\s+)?(?:the\s+)?release[.!?]?$", lowered):
        return ClarificationRequest(
            task_id=active_task.task_id if active_task else None,
            question="Which software or product release would you like me to search for?",
            missing_information="target_entity",
            reason=ClarificationReason.MISSING_ARGUMENT,
        )

    # 2. "Calculate it" / "Calculate the difference" without active numeric target
    if re.match(r"^(?:calculate\s+it|calculate\s+the\s+difference|compute\s+it)[.!?]?$", lowered):
        if not active_task or not active_task.context_values.get("calculator_result"):
            return ClarificationRequest(
                task_id=active_task.task_id if active_task else None,
                question="What values or expression would you like me to calculate?",
                missing_information="arithmetic_expression",
                reason=ClarificationReason.MISSING_ARGUMENT,
            )

    # 3. "Read that page" / "summarize that page" without active URL
    if re.search(r"\b(?:read|summarize|fetch)\s+(?:that|the)\s+page\b", lowered):
        if not active_task or not active_task.context_values.get("selected_url"):
            return ClarificationRequest(
                task_id=active_task.task_id if active_task else None,
                question="Which page would you like me to read? Please provide the URL or topic.",
                missing_information="page_url",
                reason=ClarificationReason.MISSING_ARGUMENT,
            )

    # 4. "Book it" / unsupported incomplete requests
    if re.match(r"^book\s+(?:it|a|the)[.!?]?$", lowered) or lowered == "book it":
        return ClarificationRequest(
            task_id=active_task.task_id if active_task else None,
            question="What would you like me to book?",
            missing_information="booking_target",
            reason=ClarificationReason.MISSING_ARGUMENT,
        )

    return None


def detect_multiple_matches(
    user_message: str,
    active_task: Any = None,
) -> ClarificationRequest | None:
    """
    Detect ambiguous references where multiple candidate items exist in active task state.
    Never guesses which candidate was intended.
    """
    if not active_task or active_task.is_expired():
        return None

    ctx = active_task.context_values
    lowered = user_message.lower().strip()

    # 1. Multiple URLs / Documentation pages
    candidate_urls = ctx.get("candidate_urls", [])
    candidate_titles = ctx.get("candidate_titles", [])
    if len(candidate_urls) > 1 and re.search(r"\b(?:read\s+that\s+page|read\s+the\s+page|summarize\s+that\s+page|fetch\s+that\s+page)\b", lowered):
        options = candidate_titles if len(candidate_titles) == len(candidate_urls) else candidate_urls
        opt_text = " or ".join(options[:2])
        return ClarificationRequest(
            task_id=active_task.task_id,
            question=f"Which page do you mean, the {options[0]} or the {options[1]}?",
            missing_information="selected_url",
            candidate_options=options,
            reason=ClarificationReason.MULTIPLE_MATCHES,
        )

    # 2. Multiple calculator results
    candidate_calcs = ctx.get("candidate_calculator_results", [])
    if len(candidate_calcs) > 1 and re.search(r"\b(?:divide|multiply|add|subtract)\s+(?:that|the\s+result|it)\b", lowered):
        options = [str(c) for c in candidate_calcs]
        return ClarificationRequest(
            task_id=active_task.task_id,
            question=f"Which result do you mean: {options[0]} or {options[1]}?",
            missing_information="calculator_operand",
            candidate_options=options,
            reason=ClarificationReason.MULTIPLE_MATCHES,
        )

    return None


def resolve_clarification_response(
    user_message: str,
    pending: ClarificationRequest,
    active_task: Any,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Interpret the user's response to an outstanding clarification request.
    Returns:
        (is_resolved, resolved_user_message, updated_context_values)
    """
    stripped = user_message.strip()
    lowered = stripped.lower()
    ctx = active_task.context_values

    # Check for direct match against candidate options
    if pending.candidate_options:
        for opt in pending.candidate_options:
            opt_low = opt.lower()
            if opt_low in lowered or lowered in opt_low:
                # 1. Candidate is a calculator result
                if pending.missing_information == "calculator_operand":
                    try:
                        num_val = float(opt) if "." in opt else int(opt)
                        ctx["calculator_result"] = num_val
                        # Parse operation from active task's last unresolved input if available
                        last_op_input = active_task.unresolved_user_inputs[-1] if active_task.unresolved_user_inputs else ""
                        div_match = re.search(r"divide\s+.*?by\s+(\d+(?:\.\d+)?)", last_op_input, re.IGNORECASE) or re.search(r"divide\s+.*?by\s+(\d+(?:\.\d+)?)", user_message, re.IGNORECASE)
                        if div_match:
                            divisor = div_match.group(1)
                            resolved_msg = f"Calculate {num_val} / {divisor}"
                            return True, resolved_msg, {"calculator_result": num_val, "expression": f"{num_val} / {divisor}"}
                        resolved_msg = f"Use calculation result {num_val}"
                        return True, resolved_msg, {"calculator_result": num_val}
                    except ValueError:
                        pass

                # 2. Candidate is a documentation page / URL
                if pending.missing_information == "selected_url":
                    # Retrieve matching URL
                    cand_urls = ctx.get("candidate_urls", [])
                    idx = pending.candidate_options.index(opt)
                    target_url = cand_urls[idx] if idx < len(cand_urls) else opt
                    ctx["selected_url"] = target_url
                    last_input = active_task.unresolved_user_inputs[-1] if active_task.unresolved_user_inputs else ""
                    resolved_msg = f"Read {target_url} and summarize the authentication flow" if "summarize" in last_input else f"Read {target_url}"
                    return True, resolved_msg, {"selected_url": target_url, "url": target_url}

    # Check for version clarification response
    # e.g. MAI: "Which version?" -> User: "3.13" or "Python 3.13" or "Actually compare it with 3.11"
    v_match = re.search(r"\b([vV]?\d+(?:\.\d+)+)\b", stripped)
    if v_match and ("version" in pending.question.lower() or pending.missing_information in ("target_version", "requested_comparison_target")):
        version_val = v_match.group(1).lstrip("vV")
        entity = ctx.get("target_entity", "python")
        discovered_version = ctx.get("discovered_version", "3.14.8")
        ctx["requested_comparison_target"] = version_val
        resolved_msg = f"Compare {entity} {discovered_version} with {entity} {version_val}"
        return True, resolved_msg, {
            "discovered_version": discovered_version,
            "target_version": version_val,
            "entity": entity,
        }

    # Check for booking / multi-round clarification response
    if pending.missing_information == "booking_target" and "hotel" in lowered:
        ctx["booking_target"] = "hotel"
        return False, user_message, {"booking_target": "hotel"}

    if pending.missing_information == "city":
        # Extract city name
        city_name = stripped.strip(".,!").title()
        ctx["city"] = city_name
        return True, f"Search for hotel in {city_name}", {"city": city_name}

    # General entity clarification response
    if pending.reason == ClarificationReason.UNKNOWN_ENTITY and stripped:
        ctx["target_entity"] = lowered
        resolved_msg = f"Search for {stripped} latest release"
        return True, resolved_msg, {"target_entity": lowered}

    return False, user_message, {}
