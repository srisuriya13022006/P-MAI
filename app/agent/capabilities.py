"""
P10 — Tool Capabilities and Metadata.
Exposes structured metadata, capability requirements, and deterministic candidate scoring for planning.
"""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from app.tools.registry import ToolRegistry


class PlannerCapability(str, Enum):
    """Explicit tool capability requirements that decompose user requests."""

    CURRENT_INFORMATION = "CURRENT_INFORMATION"
    PUBLIC_WEB_INFORMATION = "PUBLIC_WEB_INFORMATION"
    SPECIFIC_WEB_PAGE_CONTENT = "SPECIFIC_WEB_PAGE_CONTENT"
    NUMERIC_CALCULATION = "NUMERIC_CALCULATION"
    DATE_TIME_LOOKUP = "DATE_TIME_LOOKUP"
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    FILE_SEARCH = "FILE_SEARCH"
    USER_CONTEXT = "USER_CONTEXT"
    TASK_STATE = "TASK_STATE"
    VERSION_EXTRACTION = "VERSION_EXTRACTION"
    NUMERIC_COMPARISON = "NUMERIC_COMPARISON"
    DATE_OFFSET_CALCULATION = "DATE_OFFSET_CALCULATION"


@dataclass
class ToolCapability:
    """Explicit capability metadata for a tool."""

    name: str
    description: str
    capability_tags: list[PlannerCapability | str] = field(default_factory=list)
    required_arguments: list[str] = field(default_factory=list)
    argument_schema: dict[str, Any] = field(default_factory=dict)
    is_read_only: bool = True
    mutates_state: bool = False
    requires_confirmation: bool = False
    supported_input_types: list[str] = field(default_factory=list)
    output_type: str = "text"
    supported_downstream_consumers: list[str] = field(default_factory=list)

    @property
    def tool_name(self) -> str:
        return self.name

    @property
    def read_write_behavior(self) -> str:
        return "read_only" if self.is_read_only else "state_mutating"

    @property
    def accepted_input_types(self) -> list[str]:
        return self.supported_input_types

    @property
    def produced_output_type(self) -> str:
        return self.output_type

    @property
    def confirmation_requirement(self) -> bool:
        return self.requires_confirmation


# Built-in tool capability catalog
BUILTIN_CAPABILITIES: dict[str, ToolCapability] = {
    "calculator": ToolCapability(
        name="calculator",
        description="Perform arithmetic calculations and version/date offset arithmetic.",
        capability_tags=[
            PlannerCapability.NUMERIC_CALCULATION,
            PlannerCapability.NUMERIC_COMPARISON,
            PlannerCapability.DATE_OFFSET_CALCULATION,
        ],
        required_arguments=["expression"],
        argument_schema={"expression": "str (mathematical expression)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none", "numeric_value", "version_info", "date_offset_result"],
        output_type="numeric_value",
        supported_downstream_consumers=["calculator"],
    ),
    "datetime": ToolCapability(
        name="datetime",
        description="Retrieve current date, time, day of week, or timezone information.",
        capability_tags=[PlannerCapability.DATE_TIME_LOOKUP],
        required_arguments=[],
        argument_schema={"location": "str (optional)", "query_type": "str (optional)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none"],
        output_type="datetime_result",
        supported_downstream_consumers=["calculator", "web_search"],
    ),
    "web_search": ToolCapability(
        name="web_search",
        description="Search the public web for real-time information, releases, or news.",
        capability_tags=[
            PlannerCapability.CURRENT_INFORMATION,
            PlannerCapability.PUBLIC_WEB_INFORMATION,
        ],
        required_arguments=["query"],
        argument_schema={"query": "str (search query terms)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none", "datetime_result"],
        output_type="web_search_results",
        supported_downstream_consumers=["web_fetch", "calculator"],
    ),
    "web_fetch": ToolCapability(
        name="web_fetch",
        description="Fetch and extract text content from a specific web URL.",
        capability_tags=[PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT],
        required_arguments=["url"],
        argument_schema={"url": "str (http/https URL)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none", "web_search_results"],
        output_type="web_page_content",
        supported_downstream_consumers=["calculator", "remember_memory"],
    ),
    "file_search": ToolCapability(
        name="file_search",
        description="Search for files and code documents within the local workspace.",
        capability_tags=[PlannerCapability.FILE_SEARCH],
        required_arguments=["query"],
        argument_schema={"query": "str (search keyword)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none"],
        output_type="file_search_results",
        supported_downstream_consumers=[],
    ),
    "remember_memory": ToolCapability(
        name="remember_memory",
        description="Store user preferences, facts, or instructions in long-term memory.",
        capability_tags=[PlannerCapability.MEMORY_WRITE],
        required_arguments=["content"],
        argument_schema={"content": "str (information to remember)"},
        is_read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        supported_input_types=["memory_candidate"],
        output_type="memory_record",
        supported_downstream_consumers=[],
    ),
    "search_memory": ToolCapability(
        name="search_memory",
        description="Search previously stored user memories.",
        capability_tags=[PlannerCapability.MEMORY_READ],
        required_arguments=["query"],
        argument_schema={"query": "str (search query)"},
        is_read_only=True,
        mutates_state=False,
        requires_confirmation=False,
        supported_input_types=["none"],
        output_type="memory_record",
        supported_downstream_consumers=[],
    ),
    "update_memory": ToolCapability(
        name="update_memory",
        description="Update an existing stored user memory.",
        capability_tags=[PlannerCapability.MEMORY_WRITE],
        required_arguments=["memory_id", "content"],
        argument_schema={"memory_id": "int", "content": "str"},
        is_read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        supported_input_types=["memory_candidate"],
        output_type="memory_record",
        supported_downstream_consumers=[],
    ),
    "forget_memory": ToolCapability(
        name="forget_memory",
        description="Delete a stored memory by query or ID.",
        capability_tags=[PlannerCapability.MEMORY_WRITE],
        required_arguments=["query"],
        argument_schema={"query": "str"},
        is_read_only=False,
        mutates_state=True,
        requires_confirmation=True,
        supported_input_types=["memory_candidate"],
        output_type="memory_record",
        supported_downstream_consumers=[],
    ),
}

# Explicitly allowed dependency combinations: (predecessor_tool, consumer_tool) -> set of allowed extractors
ALLOWED_TOOL_DEPENDENCIES: dict[tuple[str, str], set[str | None]] = {
    ("web_search", "calculator"): {"extract_version_comparison"},
    ("datetime", "calculator"): {"extract_date_offset"},
    ("web_fetch", "remember_memory"): {None},  # authorized memory write only with explicit user intent
    ("web_search", "web_fetch"): {"extract_search_url"},
    ("web_fetch", "calculator"): {"extract_version_comparison", "extract_page_calculation"},
    ("datetime", "web_search"): {"extract_date_for_search"},
    ("calculator", "calculator"): {"extract_chained_calculation"},
}


def decompose_request_capabilities(
    user_message: str,
    active_task: Any = None,
) -> list[PlannerCapability]:
    """
    Decompose a user request into an explicit, deterministic list of required capabilities.
    """
    lowered = user_message.lower().strip()
    caps: list[PlannerCapability] = []

    # 1. Direct URL provided in request
    url_match = bool(re.search(r"https?://[^\s>]+", user_message))
    if url_match:
        caps.append(PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT)

    # 2. Reusing active task URL
    has_active_url = bool(
        active_task
        and not active_task.is_expired()
        and active_task.context_values.get("selected_url")
    )
    is_task_url_reference = any(
        kw in lowered
        for kw in ("that page", "the same page", "summarize that page", "summarize the page", "summarize it")
    )
    if has_active_url and is_task_url_reference and not url_match:
        caps.append(PlannerCapability.TASK_STATE)
        caps.append(PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT)

    # 3. Numeric calculation
    math_expr = bool(re.search(r"\d+\s*[\+\-\*/%]\s*\d+", user_message))
    math_words = any(
        kw in lowered
        for kw in ("calculate", "multiply", "multiplied", "divide", "divided", "subtract", "minus", "plus", "times", "percent")
    )
    if math_expr or math_words:
        caps.append(PlannerCapability.NUMERIC_CALCULATION)

    # 4. Version comparison
    has_version_comp = any(
        kw in lowered
        for kw in ("compare", "comparison", "newer than", "older than", "how many minor versions", "how many versions", "difference between", "difference with")
    ) and bool(re.search(r"[vV]?\d+(?:\.\d+)+", user_message))
    if has_version_comp:
        if PlannerCapability.NUMERIC_CALCULATION not in caps:
            caps.append(PlannerCapability.NUMERIC_CALCULATION)
        caps.append(PlannerCapability.NUMERIC_COMPARISON)

    # 5. Date / Time lookup
    has_datetime_query = any(
        kw in lowered
        for kw in ("what time", "what's the time", "current time", "time in", "today's date", "what date", "current date", "what day", "today")
    )
    if has_datetime_query:
        caps.append(PlannerCapability.DATE_TIME_LOOKUP)

    # 6. Date offset calculation
    has_offset = bool(
        re.search(r"\b\d+\s+days?\s+(?:after|from|before|prior to|ago)\b", lowered)
    )
    if has_offset and has_datetime_query:
        caps.append(PlannerCapability.DATE_OFFSET_CALCULATION)

    # 7. Web search
    needs_search = any(
        kw in lowered
        for kw in ("search the web", "search for", "find public info", "latest", "recent updates", "recent news", "breaking news", "documentation", "docs")
    )
    if needs_search and not url_match and not (has_active_url and is_task_url_reference):
        if PlannerCapability.CURRENT_INFORMATION not in caps:
            caps.append(PlannerCapability.CURRENT_INFORMATION)
        if PlannerCapability.PUBLIC_WEB_INFORMATION not in caps:
            caps.append(PlannerCapability.PUBLIC_WEB_INFORMATION)

        if has_version_comp:
            caps.append(PlannerCapability.VERSION_EXTRACTION)

        # Search followed by fetch
        has_fetch_instruction = any(
            kw in lowered
            for kw in ("read the page", "read the documentation", "fetch the page", "read the top result", "read it", "summarize it", "summarize the page")
        )
        if has_fetch_instruction and PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT not in caps:
            caps.append(PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT)

    # 8. File search
    has_file_search = any(
        kw in lowered
        for kw in ("project files", "search the project", "in the repository", "in the workspace", "onboarding note")
    )
    if has_file_search:
        caps.append(PlannerCapability.FILE_SEARCH)

    # 9. Memory read / write
    if lowered.startswith(("remember that ", "remember ")):
        caps.append(PlannerCapability.MEMORY_WRITE)
    elif any(kw in lowered for kw in ("search my memories", "search memory", "what do you remember")):
        caps.append(PlannerCapability.MEMORY_READ)

    return caps


def score_tool_candidates(
    required_capabilities: list[PlannerCapability],
    user_message: str,
    active_task: Any = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Score and rank tool candidates deterministically based on capability match,
    argument availability, safety, and over-planning avoidance.
    Returns: list of (tool_name, score, reason_dict) sorted by score descending.
    """
    lowered = user_message.lower()
    has_direct_url = bool(re.search(r"https?://[^\s>]+", user_message))
    has_task_url = bool(
        active_task
        and not active_task.is_expired()
        and active_task.context_values.get("selected_url")
    )

    scored: list[tuple[str, float, dict[str, Any]]] = []

    for name, tool_cap in BUILTIN_CAPABILITIES.items():
        score = 0.0
        reasons: dict[str, Any] = {}

        # 1. Capability tag overlap
        matching_tags = [tag for tag in tool_cap.capability_tags if tag in required_capabilities]
        reasons["matched_tags"] = [t.value if hasattr(t, "value") else str(t) for t in matching_tags]
        score += len(matching_tags) * 10.0

        # 2. Tool-specific argument and relevance scoring
        if name == "web_fetch":
            if has_direct_url:
                score += 15.0
                reasons["argument_availability"] = "direct URL provided in user message"
            elif has_task_url and any(kw in lowered for kw in ("that page", "the same page", "summarize it")):
                score += 15.0
                reasons["argument_availability"] = "URL available in active task state"
            elif PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT in required_capabilities:
                score += 5.0
                reasons["argument_availability"] = "dependent on upstream search result"
            else:
                score -= 10.0
                reasons["argument_penalty"] = "no direct or task URL available"

        elif name == "web_search":
            if has_direct_url:
                score -= 20.0
                reasons["redundancy_penalty"] = "direct URL already provided; web_search unnecessary"
            elif has_task_url and any(kw in lowered for kw in ("that page", "the same page", "summarize that page")):
                score -= 20.0
                reasons["redundancy_penalty"] = "active task URL exists; re-searching unnecessary"
            elif PlannerCapability.PUBLIC_WEB_INFORMATION in required_capabilities:
                score += 10.0
                reasons["primary_capability"] = "public web search required"

        elif name == "calculator":
            if PlannerCapability.NUMERIC_CALCULATION in required_capabilities or PlannerCapability.NUMERIC_COMPARISON in required_capabilities:
                score += 10.0
                reasons["primary_capability"] = "calculation / comparison requested"

        elif name == "datetime":
            if PlannerCapability.DATE_TIME_LOOKUP in required_capabilities:
                score += 10.0
                reasons["primary_capability"] = "current date/time requested"

        elif name == "remember_memory":
            if PlannerCapability.MEMORY_WRITE in required_capabilities:
                score += 20.0
                reasons["explicit_intent"] = "user explicitly requested to remember"
            else:
                score -= 50.0
                reasons["safety_penalty"] = "unauthorized state mutation without explicit remember intent"

        scored.append((name, score, reasons))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def match_capabilities_to_plan(
    required_capabilities: list[PlannerCapability],
    user_message: str,
    active_task: Any = None,
) -> list[str]:
    """
    Deterministically map required capabilities to a minimal, non-redundant tool sequence.
    Guarantees no over-planning.
    """
    has_direct_url = bool(re.search(r"https?://[^\s>]+", user_message))
    has_task_url = bool(
        active_task
        and not active_task.is_expired()
        and active_task.context_values.get("selected_url")
    )
    lowered = user_message.lower()

    # 1. Direct web fetch (URL already provided in message or active task)
    if has_direct_url:
        if PlannerCapability.NUMERIC_COMPARISON in required_capabilities:
            return ["web_fetch", "calculator"]
        return ["web_fetch"]

    if has_task_url and any(kw in lowered for kw in ("that page", "the same page", "summarize that page", "summarize it")):
        if PlannerCapability.NUMERIC_COMPARISON in required_capabilities:
            return ["web_fetch", "calculator"]
        return ["web_fetch"]

    # 2. Pure single-tool requests
    if required_capabilities == [PlannerCapability.NUMERIC_CALCULATION]:
        return ["calculator"]
    if required_capabilities == [PlannerCapability.DATE_TIME_LOOKUP]:
        return ["datetime"]
    if required_capabilities == [PlannerCapability.FILE_SEARCH]:
        return ["file_search"]
    if required_capabilities == [PlannerCapability.MEMORY_READ]:
        return ["search_memory"]
    if required_capabilities == [PlannerCapability.MEMORY_WRITE]:
        return ["remember_memory"]

    # 3. Date lookup + offset calculation
    if PlannerCapability.DATE_TIME_LOOKUP in required_capabilities and PlannerCapability.DATE_OFFSET_CALCULATION in required_capabilities:
        return ["datetime", "calculator"]

    # 4. Date lookup + Web search
    if PlannerCapability.DATE_TIME_LOOKUP in required_capabilities and PlannerCapability.PUBLIC_WEB_INFORMATION in required_capabilities:
        return ["datetime", "web_search"]

    # 5. Multi-tool web search pipelines
    if PlannerCapability.PUBLIC_WEB_INFORMATION in required_capabilities or PlannerCapability.CURRENT_INFORMATION in required_capabilities:
        # Search -> Fetch -> Calculator
        if PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT in required_capabilities and (PlannerCapability.NUMERIC_COMPARISON in required_capabilities or PlannerCapability.NUMERIC_CALCULATION in required_capabilities):
            return ["web_search", "web_fetch", "calculator"]

        # Search -> Fetch
        if PlannerCapability.SPECIFIC_WEB_PAGE_CONTENT in required_capabilities:
            return ["web_search", "web_fetch"]

        # Search -> Calculator
        if PlannerCapability.NUMERIC_COMPARISON in required_capabilities or PlannerCapability.NUMERIC_CALCULATION in required_capabilities:
            return ["web_search", "calculator"]

        # Search only
        return ["web_search"]

    # 6. Direct version comparison without search
    if PlannerCapability.NUMERIC_COMPARISON in required_capabilities:
        return ["calculator"]

    # 7. No tool needed (local knowledge, e.g. "Explain quantum computing")
    return []


def get_tool_capability(tool_name: str, registry: ToolRegistry | None = None) -> ToolCapability | None:
    """Get the capability metadata for a tool."""
    if tool_name in BUILTIN_CAPABILITIES:
        return BUILTIN_CAPABILITIES[tool_name]

    if registry and registry.has(tool_name):
        tool = registry.get(tool_name)
        req_args: list[str] = []
        if hasattr(tool, "input_schema") and tool.input_schema is not None:
            fields = getattr(tool.input_schema, "model_fields", {})
            req_args = [name for name, f in fields.items() if f.is_required()]
        return ToolCapability(
            name=tool.name,
            description=tool.description,
            required_arguments=req_args,
            is_read_only=True,
            mutates_state=False,
            requires_confirmation=False,
            supported_input_types=["any"],
            output_type="custom",
        )

    return None


def validate_capability_metadata_consistency(
    registry: ToolRegistry,
    require_all_dependencies: bool = False,
) -> tuple[bool, list[str]]:
    """
    Ensure BUILTIN_CAPABILITIES and ALLOWED_TOOL_DEPENDENCIES cannot silently drift
    from actual ToolRegistry registrations.
    """
    errors: list[str] = []

    # 1. Registered tools without explicit capability metadata in catalog
    for tool_name in registry._tools:
        if tool_name not in BUILTIN_CAPABILITIES:
            errors.append(f"Registered tool '{tool_name}' is missing capability metadata in catalog.")

    # 2. Dependency pairs referencing tools not in catalog
    for (pred, cons), _ in ALLOWED_TOOL_DEPENDENCIES.items():
        if pred not in BUILTIN_CAPABILITIES and not registry.has(pred):
            errors.append(f"Allowed tool dependency predecessor '{pred}' is unknown.")
        if cons not in BUILTIN_CAPABILITIES and not registry.has(cons):
            errors.append(f"Allowed tool dependency consumer '{cons}' is unknown.")
        if require_all_dependencies:
            if not registry.has(pred):
                errors.append(f"Allowed tool dependency predecessor '{pred}' is not registered in ToolRegistry.")
            if not registry.has(cons):
                errors.append(f"Allowed tool dependency consumer '{cons}' is not registered in ToolRegistry.")

    return len(errors) == 0, errors
