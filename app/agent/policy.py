import re

from app.agent.tool_argument_resolver import (
    resolve_datetime_arguments,
    resolve_web_fetch_arguments,
    resolve_web_search_arguments,
)
from app.schemas.agent import AgentDecision


def is_self_identity_query(user_message: str) -> bool:
    """
    Detect whether a query is asking about MAI's self-identity, name, nature, or capabilities.
    Must explicitly distinguish MAI self-identity from queries about the user's identity/memories.
    """
    lowered = user_message.lower().strip()

    # 1. First check if it contains explicit USER identity/memory markers
    user_memory_markers = (
        "my name",
        "about me",
        "about myself",
        "who am i",
        "my goal",
        "my current goal",
        "my project",
        "my preference",
        "my preferred",
        "what do i",
        "do i have",
        "what i ",
    )
    if any(marker in lowered for marker in user_memory_markers):
        return False

    # 2. Strip leading greetings, farewells, conversational interjections, and assistant names
    cleaned = re.sub(
        r"^(?:(?:hi|hey|hello|greetings|good\s+(?:morning|afternoon|evening)|bye|goodbye|see\s+ya|farewell|ok|okay|please|mai|hey\s+mai)\b[\s,\.!?-]*)+",
        "",
        lowered,
        flags=re.IGNORECASE,
    ).strip()

    if any(marker in cleaned for marker in user_memory_markers):
        return False

    # 3. Check for self-identity patterns
    self_identity_patterns = (
        # Name
        r"\b(?:what(?:s|\s+is|'s|\s+was)?\s+your\s+name|tell\s+me\s+your\s+name|do\s+you\s+have\s+a\s+name|your\s+name)\b",
        # Who / What are you
        r"\b(?:who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|what\s+r\s+u|who\s+made\s+you|who\s+created\s+you)\b",
        # About MAI
        r"\b(?:tell\s+me\s+about\s+yourself|describe\s+yourself|introduce\s+yourself)\b",
        # Nature
        r"\b(?:are\s+you\s+(?:an?\s+)?(?:ai|robot|bot|human|machine|program))\b",
        # Model
        r"\b(?:which|what)\s+(?:model|llm)\s+(?:are\s+you|is\s+this)\b",
        # Capabilities
        r"\b(?:what\s+can\s+you\s+do|what\s+are\s+your\s+capabilities|how\s+can\s+you\s+help(?:\s+me)?)\b",
    )

    return any(bool(re.search(pat, cleaned)) for pat in self_identity_patterns)


def is_explicit_memory_write_request(message: str) -> tuple[bool, str]:
    """Detect explicit commands to remember or save personal memory."""
    m = re.match(
        r"^(?:please\s+)?remember(?:\s+that)?\s+(.+)|^(?:please\s+)?save\s+(?:this\s+preference:?\s*|this:?\s*|that:?\s*|that\s+preference:?\s*)(.+)",
        message.strip(),
        re.IGNORECASE,
    )
    if m:
        content = (m.group(1) or m.group(2) or "").strip().rstrip(".!?")
        return True, content
    return False, ""


def apply_policy(
    user_message: str,
    decision: AgentDecision,
) -> AgentDecision:
    """Apply deterministic MAI routing and safety rules."""

    lowered = user_message.lower().strip()

    # 0. Check for MAI self-identity queries (e.g. "what is your name?", "who are you?", "bye, what is your name?")
    if is_self_identity_query(user_message) or (
        getattr(decision, "intent", None) == "identify_self"
        and not any(m in lowered for m in ("my name", "about me", "about myself", "who am i", "my project"))
    ):
        if hasattr(decision, "intent"):
            decision.intent = "identify_self"
        decision.route = "local"
        decision.tools = []
        decision.tool_arguments = {}
        decision.needs_clarification = False
        decision.reason = "Conversational self-identity query answered locally without tools or memory."
        return decision

    # Explicit check for user's own name query
    if re.search(r"\b(?:what\s+(?:is|'s)|do\s+you\s+(?:know|remember))\s+my\s+name\b", lowered):
        if hasattr(decision, "intent"):
            decision.intent = "personal_memory_retrieval"
        decision.route = "memory"
        decision.tools = []
        decision.tool_arguments = {}
        decision.needs_clarification = False
        decision.reason = "Questions about the user's name use memory retrieval."
        return decision

    has_math_expression = bool(
        re.search(
            r"\d\s*[\+\-\*/%]\s*\d",
            lowered,
        )
    )

    arithmetic_keywords = (
        "calculate",
        "multiply",
        "multiplied",
        "divide",
        "divided",
        "plus",
        "minus",
        "subtract",
        "add",
        "times",
        "percent",
    )

    datetime_keywords = (
        "what time",
        "what's the time",
        "time in",
        "current time",
        "time is it",
        "today's date",
        "todays date",
        "what date",
        "what's the date",
        "what is the date",
        "date in",
        "today",
        "tomorrow",
        "yesterday",
        "what day",
        "what's the day",
        "what is the day",
        "day in",
        "current date",
        "current datetime",
        "date and time",
        "day is today",
        "day is it",
    )

    contains_arithmetic_keyword = any(
        keyword in lowered
        for keyword in arithmetic_keywords
    )

    contains_datetime_keyword = any(
        keyword in lowered
        for keyword in datetime_keywords
    )

    is_exp_mem, exp_content = is_explicit_memory_write_request(user_message)
    if is_exp_mem:
        decision.intent = "remember_memory"
        decision.route = "tool"
        decision.tools = ["remember_memory"]
        decision.tool_arguments = {
            "content": exp_content,
        }
        decision.needs_clarification = False
        decision.reason = "Explicit memory requests use the remember_memory tool."
        return decision

    if lowered.startswith(("forget that ", "forget ")):
        decision.intent = "forget_memory"
        decision.route = "tool"
        decision.tools = ["forget_memory"]
        decision.tool_arguments = {
            "query": re.sub(r"^forget(?: that)?\s+", "", user_message, flags=re.IGNORECASE).rstrip(".!?")
        }
        decision.needs_clarification = False
        decision.reason = "Explicit memory requests use the forget_memory tool."
        return decision

    update_match = re.match(
        r"update memory\s+(\d+)\s+(?:to|as)\s+(.+)",
        user_message,
        re.IGNORECASE,
    )
    if update_match:
        decision.intent = "update_memory"
        decision.route = "tool"
        decision.tools = ["update_memory"]
        decision.tool_arguments = {
            "memory_id": int(update_match.group(1)),
            "content": update_match.group(2).rstrip(".!?").strip(),
        }
        decision.needs_clarification = False
        decision.reason = "Explicit memory requests use the update_memory tool."
        return decision

    broad_profile_phrases = (
        "what do you know about me",
        "what do you remember about me",
        "tell me about myself",
        "what have you learned about me",
        "what do you remember",
        "what do you know",
        "who am i",
    )
    if any(phrase in lowered for phrase in broad_profile_phrases):
        decision.intent = "broad_profile_query"
        decision.route = "memory"
        decision.tools = []
        decision.tool_arguments = {}
        decision.needs_clarification = False
        decision.reason = "Broad user profile questions use memory profile context and conversational synthesis."
        return decision

    if any(phrase in lowered for phrase in ("search my memories", "search memory", "find memory")):
        decision.intent = "search_memory"
        decision.route = "tool"
        decision.tools = ["search_memory"]
        search_query = re.sub(
            r"^(?:search\s+(?:my\s+)?memories(?:\s+for)?|search\s+memory(?:\s+for)?|find\s+memory(?:\s+for)?)\s*",
            "",
            user_message,
            flags=re.IGNORECASE,
        ).rstrip(".!?").strip()
        decision.tool_arguments = {"query": search_query or user_message}
        decision.needs_clarification = False
        decision.reason = "Explicit memory search requests use the search_memory tool."
        return decision

    expression_match = re.search(
        r"\d+\s*[\+\-\*/%]\s*\d+",
        lowered,
    )
    extracted_expression = (
        expression_match.group(0) if expression_match else ""
    )

    location_match = re.search(
        r"\bin\s+([a-z][a-z\s\-]+?)(?:\?|\.|!|$)",
        lowered,
    )
    extracted_location = (
        location_match.group(1).strip().title()
        if location_match
        else None
    )

    current_info_keywords = (
        "latest",
        "recent",
        "current news",
        "breaking news",
        "latest updates",
        "recent updates",
        "recent research",
        "recent announcement",
        "recent announcements",
        "this week",
        "this month",
        "what happened today",
        "news today",
    )
    explicit_web_search_phrases = (
        "search the web",
        "search for",
        "find public info",
        "look up",
        "google",
        "public profile",
        "what is the latest on",
        "what's the latest on",
    )
    private_info_markers = (
        "phone number",
        "email address",
        "home address",
        "personal details",
        "student record",
        "private info",
    )
    personal_statement_starters = (
        "i am ",
        "i'm ",
        "i prefer",
        "i no longer",
        "i like",
        "i dislike",
        "my name",
        "my goal",
        "my current goal",
        "my project",
    )
    personal_query_starters = (
        "what do i ",
        "what is my ",
        "what are my ",
        "who am i",
        "do i ",
    )
    is_personal_statement = any(
        lowered.startswith(prefix) for prefix in personal_statement_starters
    )
    is_personal_query = (
        is_personal_statement
        or any(lowered.startswith(prefix) for prefix in personal_query_starters)
    )
    explicit_search_intent = (
        any(phrase in lowered for phrase in explicit_web_search_phrases)
        or bool(re.search(r"\bsearch\s+([a-zA-Z0-9\.\-]+\.[a-zA-Z]{2,63})\s+(?:for|about|on)?\b", lowered))
        or bool(re.search(r"\bsite:[a-zA-Z0-9\.\-]+\b", lowered))
        or bool(
            re.search(r"\b(?:find|search)\b.*\b(?:documentation|docs)\b", lowered)
            and not any(
                kw in lowered
                for kw in ("in project", "project files", "in workspace", "in repo", "in repository", "in the files")
            )
        )
    )
    needs_current_info = any(
        keyword in lowered for keyword in current_info_keywords
    )

    # Direct URL page retrieval requests (e.g. "Read https://...", "Explain https://...")
    url_match = re.search(r"https?://[^\s>]+", user_message)
    if url_match:
        read_keywords = (
            "read",
            "fetch",
            "open",
            "summarize",
            "explain",
            "analyze",
            "inspect",
            "what does",
            "what is on",
            "content of",
            "tell me about what",
        )
        wants_to_read = any(kw in lowered for kw in read_keywords) or lowered.startswith("http")
        if wants_to_read and not explicit_search_intent:
            if re.search(r"\bremember(?:\s+that)?\s+", user_message, re.IGNORECASE):
                decision.intent = "multi_tool"
                decision.route = "tool"
                decision.tools = ["web_fetch", "remember_memory"]
                decision.tool_arguments = resolve_web_fetch_arguments(
                    user_message,
                    {"url": url_match.group(0).rstrip(".,;!?'\")")},
                )
                rem_match = re.search(r"\bremember(?:\s+that)?\s+(.+)$", user_message, re.IGNORECASE)
                if rem_match:
                    decision.tool_arguments["content"] = rem_match.group(1).rstrip(".!?").strip()
                decision.needs_clarification = False
                decision.reason = "Web page retrieval with memory storage uses web_fetch and remember_memory."
                return decision

            decision.intent = "web_fetch"
            decision.route = "tool"
            decision.tools = ["web_fetch"]
            decision.tool_arguments = resolve_web_fetch_arguments(
                user_message,
                {"url": url_match.group(0).rstrip(".,;!?'\")")},
            )
            decision.needs_clarification = False
            decision.reason = "Direct web page retrieval requested for provided URL."
            return decision

    has_search_comparison = (needs_current_info or explicit_search_intent) and any(
        kw in lowered
        for kw in (
            "compare",
            "newer than",
            "older than",
            "how many minor versions",
            "how many versions",
            "difference between",
            "difference with",
        )
    )
    has_fetch_instruction = any(
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

    # 3-step chain: web_search -> web_fetch -> calculator
    if (needs_current_info or explicit_search_intent) and has_fetch_instruction and (has_search_comparison or contains_arithmetic_keyword or has_math_expression):
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["web_search", "web_fetch", "calculator"]
        decision.tool_arguments = resolve_web_search_arguments(user_message, decision.tool_arguments)
        decision.needs_clarification = False
        decision.reason = "Search, page fetch, and version calculation pipeline."
        return decision

    # 2-step chain: web_search -> web_fetch
    if (needs_current_info or explicit_search_intent) and has_fetch_instruction and not url_match:
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["web_search", "web_fetch"]
        decision.tool_arguments = resolve_web_search_arguments(user_message, decision.tool_arguments)
        decision.needs_clarification = False
        decision.reason = "Search and fetch top result pipeline."
        return decision

    # 2-step chain: datetime -> web_search
    has_explicit_datetime_query = any(phrase in lowered for phrase in ("what date", "what is the date", "today's date and", "current date and", "what time is it and"))
    if has_explicit_datetime_query and any(kw in lowered for kw in ("headlines", "news", "events", "search")):
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["datetime", "web_search"]
        decision.tool_arguments = resolve_web_search_arguments(user_message, decision.tool_arguments)
        decision.needs_clarification = False
        decision.reason = "Datetime reference lookup followed by web search."
        return decision

    if has_search_comparison and not any(marker in lowered for marker in private_info_markers):
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["web_search", "calculator"]
        decision.tool_arguments = resolve_web_search_arguments(
            user_message,
            decision.tool_arguments,
        )
        decision.needs_clarification = False
        decision.reason = "Search and comparison requests use web_search and calculator in sequence."
        return decision

    if (
        (needs_current_info or explicit_search_intent)
        and not any(marker in lowered for marker in private_info_markers)
        and not contains_arithmetic_keyword
        and not (is_personal_query and not explicit_search_intent)
    ):
        decision.intent = "web_search"
        decision.route = "tool"
        decision.tools = ["web_search"]
        decision.tool_arguments = resolve_web_search_arguments(
            user_message,
            decision.tool_arguments,
        )
        decision.needs_clarification = False
        decision.reason = (
            "Requests for current public information or explicit web searches are best answered with the web search tool."
        )
        return decision

    future_offset_match = re.search(r"\b(\d+)\s+days?\s+(?:after|from|past)\s+(?:today|now)\b", lowered)
    past_offset_match = re.search(r"\b(\d+)\s+days?\s+(?:before|prior to)\s+(?:today|now)\b", lowered) or re.search(r"\b(\d+)\s+days?\s+ago\b", lowered)
    if (contains_datetime_keyword or "date" in lowered) and (future_offset_match or past_offset_match):
        days = int(future_offset_match.group(1)) if future_offset_match else -int(past_offset_match.group(1))
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["datetime", "calculator"]
        decision.tool_arguments = {
            "query_type": "date",
            "days": days,
        }
        decision.needs_clarification = False
        decision.reason = "Date offset calculation uses datetime and calculator tools in sequence."
        return decision

    if contains_datetime_keyword and (
        contains_arithmetic_keyword or has_math_expression
    ):
        decision.intent = "multi_tool"
        decision.route = "tool"
        decision.tools = ["calculator", "datetime"]
        decision.tool_arguments = {
            "expression": extracted_expression or user_message,
            "location": extracted_location,
        }
        decision.needs_clarification = False
        decision.reason = (
            "Combined arithmetic and date/time requests use "
            "the calculator and datetime tools in sequence."
        )
        return decision

    file_keywords = (
        "document",
        "documents",
        "file",
        "files",
        "folder",
        "notes",
        "readme",
        "project files",
        "search the project",
        "look in the files",
        "find in the docs",
        "in the repository",
        "in the workspace",
        "onboarding",
        "setup guide",
        "guide",
    )
    needs_file_search = any(keyword in lowered for keyword in file_keywords)

    if needs_file_search and not (
        contains_arithmetic_keyword or contains_datetime_keyword
    ):
        decision.intent = "file_search"
        decision.route = "tool"
        decision.tools = ["file_search"]
        decision.tool_arguments = {"query": user_message.strip().rstrip("?!.")}
        decision.needs_clarification = False
        decision.reason = "Project-file lookups are handled by the file_search tool."
        return decision

    if contains_datetime_keyword:
        decision.intent = "datetime"
        decision.route = "tool"
        decision.tools = ["datetime"]
        decision.tool_arguments = resolve_datetime_arguments(
            user_message,
            decision.tool_arguments,
        )
        decision.needs_clarification = False
        decision.reason = (
            "Current date and time requests are handled "
            "by the datetime tool."
        )
        return decision

    v_comp_direct = re.search(
        r"compare\s+.*?([vV]?\d+(?:\.\d+)+).*?(?:with|to)\s+.*?([vV]?\d+(?:\.\d+)+)",
        lowered,
    )
    if v_comp_direct and not (needs_current_info or explicit_search_intent):
        decision.intent = "arithmetic"
        decision.route = "tool"
        decision.tools = ["calculator"]
        decision.needs_clarification = False
        decision.reason = "Direct version comparison using calculator."
        return decision

    if contains_arithmetic_keyword or has_math_expression:
        decision.intent = "arithmetic"
        decision.route = "tool"
        decision.tools = ["calculator"]
        decision.needs_clarification = False
        decision.reason = (
            "Arithmetic requests are handled by "
            "the calculator tool."
        )
    if "remember_memory" in decision.tools and not (
        lowered.startswith(("remember that ", "remember "))
        or is_explicit_memory_write_request(user_message)[0]
    ):
        decision.tools = [t for t in decision.tools if t != "remember_memory"]
        decision.tool_arguments = {}
        if decision.route in ("tool", "memory"):
            is_retrieval_query = bool(
                lowered.endswith("?")
                or re.match(r"^(?:what|who|where|when|why|how|do\s+you|can\s+you|is\s+my|are\s+my)\b", lowered)
            )
            if is_retrieval_query:
                decision.route = "memory"
                decision.intent = "personal_memory_retrieval"
                decision.reason = "Personal memory question routes to memory retrieval, not memory writing."
            else:
                decision.route = "local"
                decision.intent = "conversational"
                decision.reason = "Personal statement without explicit remember intent routes to local conversational handling."

    if decision.route == "memory":
        is_retrieval_query = bool(
            lowered.endswith("?")
            or re.match(r"^(?:what|who|where|when|why|how|do\s+you|can\s+you|is\s+my|are\s+my|tell\s+me\s+about\s+(?:myself|me))\b", lowered)
        )
        if not is_retrieval_query:
            decision.route = "local"
            decision.intent = "conversational"
            decision.reason = "Personal statement without retrieval query routes to local conversational handling."
        else:
            decision.tools = []
            decision.tool_arguments = {}

    return decision