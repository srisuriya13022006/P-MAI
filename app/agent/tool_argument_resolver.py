import re
from typing import Any


def resolve_calculator_arguments(
    user_message: str,
) -> dict[str, str]:
    """
    Extract the arithmetic expression from a natural-language
    calculator request.
    """

    text = user_message.lower().strip()

    # Remove common conversational prefixes.
    text = re.sub(
        r"^(what is|what's|calculate|can you calculate|please calculate)\s+",
        "",
        text,
    )

    # Remove question marks.
    text = text.rstrip("?").strip()

    return {
        "expression": text,
    }


def resolve_datetime_arguments(
    user_message: str,
    existing_arguments: dict | None = None,
) -> dict[str, str]:
    """
    Extract or refine location, timezone, relative_day, and query_type
    for a datetime request.
    """
    args = dict(existing_arguments or {})
    lowered = user_message.lower().strip()

    # 1. Location extraction
    loc_match = re.search(
        r"\b(?:in|for|at)\s+([a-z][a-z\s\-]+?)(?:\?|\.|!|$)",
        lowered,
    )
    if loc_match:
        args["location"] = loc_match.group(1).strip().title()
        args.pop("timezone", None)
    elif not re.search(r"\b(?:in|for|at)\b", lowered):
        # If the user did not specify a location, purge any hallucinated or inherited location
        args.pop("location", None)
        args.pop("timezone", None)

    # 2. Relative day extraction
    if "tomorrow" in lowered:
        args["relative_day"] = "tomorrow"
    elif "yesterday" in lowered:
        args["relative_day"] = "yesterday"
    elif not args.get("relative_day"):
        args["relative_day"] = "today"

    # 3. Query type extraction
    if "date and time" in lowered or "current datetime" in lowered:
        args["query_type"] = "datetime"
    elif any(q in lowered for q in ("what time", "what's the time", "what is the time", "current time", "time is it", "time in", "the time in")):
        args["query_type"] = "time"
    elif any(q in lowered for q in ("what day", "what's the day", "what is the day", "day is today", "day is it", "what day is", "day in", "the day in", "day will tomorrow", "day was yesterday")):
        args["query_type"] = "day"
    elif any(q in lowered for q in ("what date", "what's the date", "what is the date", "today's date", "todays date", "current date", "date in", "the date in", "date is tomorrow", "date was yesterday", "date of")):
        args["query_type"] = "date"
    elif not args.get("query_type"):
        args["query_type"] = "datetime"

    return args


from app.core.config import settings
from app.tools.web.url import is_valid_domain

OFFICIAL_DOCS_MAPPING = {
    "fastapi": "fastapi.tiangolo.com",
    "python": "docs.python.org",
    "django": "docs.djangoproject.com",
    "postgresql": "postgresql.org",
    "postgres": "postgresql.org",
    "docker": "docs.docker.com",
    "kubernetes": "kubernetes.io",
    "react": "react.dev",
    "vue": "vuejs.org",
}


def resolve_web_search_arguments(
    user_message: str,
    existing_arguments: dict | None = None,
) -> dict[str, Any]:
    """Extract and clean the search query from a natural-language request,

    stripping conversational filler and search command wrappers while strictly
    preserving qualifiers (e.g. latest, recent, dates, versions, locations,
    operating systems/platforms, technical constraints, and URLs).

    Also deterministically extracts:
    - freshness constraint (e.g. 'latest', 'past_week', 'past_24_hours')
    - explicit domain restriction (e.g. 'github.com', 'site:example.com', or deterministic official docs)
    - max_results constraint (bounded between 1 and 10)
    """
    args = dict(existing_arguments or {})
    text = (user_message or "").strip()
    if not text:
        args["query"] = ""
        args.setdefault("freshness", None)
        args.setdefault("domain", None)
        args.setdefault("max_results", settings.web_search_max_results)
        return args

    lowered_original = text.lower()

    # 1. Deterministic Domain Extraction
    domain = args.get("domain")
    if not domain:
        # A. site:domain.com
        site_match = re.search(r"\bsite:([a-zA-Z0-9\.\-]+)", text, re.IGNORECASE)
        if site_match:
            candidate = site_match.group(1).lower().strip()
            if is_valid_domain(candidate):
                domain = candidate
            text = re.sub(r"\bsite:[a-zA-Z0-9\.\-]+", "", text, flags=re.IGNORECASE).strip()

    if not domain:
        # B. search domain.com for ...
        search_dom_match = re.search(
            r"\bsearch\s+([a-zA-Z0-9\.\-]+\.[a-zA-Z]{2,63})\s+(?:for|about|on)?\s*",
            text,
            re.IGNORECASE,
        )
        if search_dom_match:
            candidate = search_dom_match.group(1).lower().strip()
            if is_valid_domain(candidate):
                domain = candidate
                text = text[:search_dom_match.start()] + "search for " + text[search_dom_match.end():]

    if not domain:
        # C. ... on domain.com / from domain.com
        on_dom_match = re.search(
            r"\b(?:on|from|in)\s+([a-zA-Z0-9\.\-]+\.[a-zA-Z]{2,63})\b",
            text,
            re.IGNORECASE,
        )
        if on_dom_match:
            candidate = on_dom_match.group(1).lower().strip()
            if is_valid_domain(candidate):
                domain = candidate
                text = text[:on_dom_match.start()] + " " + text[on_dom_match.end():]

    if not domain:
        # D. Deterministic official documentation mapping
        doc_match = re.search(
            r"\bofficial\s+([a-zA-Z0-9]+)\s+(?:docs|documentation)\b",
            text,
            re.IGNORECASE,
        )
        if doc_match:
            proj = doc_match.group(1).lower()
            if proj in OFFICIAL_DOCS_MAPPING:
                domain = OFFICIAL_DOCS_MAPPING[proj]

    if domain and not is_valid_domain(domain):
        domain = None

    # 2. Deterministic Freshness Extraction
    freshness = args.get("freshness")
    if not freshness:
        if any(kw in lowered_original for kw in ("today", "past 24 hours", "last 24 hours", "in the last 24 hours")):
            freshness = "past_24_hours"
        elif "yesterday" in lowered_original:
            freshness = "yesterday"
        elif any(kw in lowered_original for kw in ("this week", "past week", "last week", "from the last week", "in the past week", "in the last week")):
            freshness = "past_week"
        elif any(kw in lowered_original for kw in ("this month", "past month", "last month", "in the past month", "in the last month")):
            freshness = "past_month"
        elif any(kw in lowered_original for kw in ("this year", "past year", "last year", "in the past year")):
            freshness = "past_year"
        elif any(kw in lowered_original for kw in ("latest", "recent", "newest", "fresh news", "breaking news")):
            freshness = "latest"

    # 3. Deterministic Max Results Extraction
    max_results = args.get("max_results")
    if max_results is None:
        count_match = re.search(
            r"\b(?:top|first|limit(?:\s+to)?|max(?:imum)?)\s+(\d{1,2})\s+(?:results|sources|links)?\b",
            text,
            re.IGNORECASE,
        )
        if not count_match:
            count_match = re.search(r"\b(\d{1,2})\s+(?:web\s+)?results\b", text, re.IGNORECASE)
        if count_match:
            try:
                cnt = int(count_match.group(1))
                if 1 <= cnt <= 10:
                    max_results = cnt
                    # Remove the count qualifier from search query text
                    text = text[:count_match.start()] + " " + text[count_match.end():]
            except ValueError:
                pass

    if max_results is None:
        max_results = getattr(settings, "web_search_max_results", 5)
    max_results = min(max(1, int(max_results)), 10)

    # Clean trailing question marks and exclamation points
    text = re.sub(r"[\?\.!]+$", "", text).strip()

    # 4. Strip polite / conversational opening salutations
    polite_openers_pattern = (
        r"^(?:"
        r"(?:hey|hi|hello)\s*[,! ]\s*"
        r"|(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        r"|please\s+(?:can\s+you\s+|could\s+you\s+)?"
        r"|i\s+(?:need|want)\s+you\s+to\s+"
        r"|(?:can\s+you\s+)?help\s+me\s+(?:to\s+)?"
        r")+"
    )
    cleaned = re.sub(polite_openers_pattern, "", text, flags=re.IGNORECASE).strip()

    # 5. Strip search command wrapper phrases
    search_wrappers_pattern = (
        r"^(?:"
        r"search\s+(?:the\s+)?web\s+(?:and\s+find|to\s+find|for)?\s*"
        r"|search\s+online\s+(?:and\s+find|to\s+find|for)?\s*"
        r"|search\s+google\s+(?:for)?\s*"
        r"|search\s+for\s+"
        r"|look\s+up\s+"
        r"|look\s+for\s+"
        r"|find\s+online\s+(?:information\s+about|info\s+about|for)?\s*"
        r"|find\s+(?:information|info|details)\s+(?:about|on|regarding)\s+"
        r"|find\s+out\s+(?:about\s+)?"
        r"|find\s+public\s+info\s+(?:about)?\s*"
        r"|find\s+"
        r"|google\s+(?:for)?\s*"
        r")+"
    )
    cleaned_after_wrapper = re.sub(search_wrappers_pattern, "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned_after_wrapper:
        cleaned = cleaned_after_wrapper

    # 6. Strip trailing temporal prepositional phrases (e.g. 'from the last week', 'in the past week')
    cleaned = re.sub(
        r"\b(?:from|in|during)\s+(?:the\s+)?(?:last|past|this)\s+(?:week|month|year|24\s+hours)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # 7. Strip residual leading connecting prepositions (e.g. "for", "about", "on")
    residual_pattern = r"^(?:for|about|on|regarding)\s+"
    cleaned = re.sub(residual_pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # 8. Strip leading article before superlatives/qualifiers or general queries if left over
    cleaned = re.sub(
        r"^(?:the|a|an)\s+(?=(?:latest|recent|current|most|official|best)\b)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # 9. Clean whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 10. Bounded length: if excessively long, preserve first 25 words
    words = cleaned.split()
    if len(words) > 25:
        cleaned = " ".join(words[:25])

    args["query"] = cleaned if cleaned else text
    args["freshness"] = freshness
    args["domain"] = domain
    args["max_results"] = max_results
    return args


def resolve_web_fetch_arguments(
    user_message: str,
    existing_arguments: dict | None = None,
) -> dict[str, str]:
    """Extract a clean, valid URL from conversational user requests."""
    args = dict(existing_arguments or {})
    if args.get("url"):
        return args

    # Extract first HTTP/HTTPS URL from user message
    match = re.search(r"https?://[^\s>]+", user_message)
    if match:
        url = match.group(0).rstrip(".,;!?'\")")
        args["url"] = url
        return args

    args["url"] = user_message.strip()
    return args