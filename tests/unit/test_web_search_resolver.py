import pytest

from app.agent.policy import apply_policy
from app.agent.tool_argument_resolver import resolve_web_search_arguments
from app.schemas.agent import AgentDecision


def _base_decision() -> AgentDecision:
    return AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )


def test_resolver_strips_search_the_web_for():
    """1. 'Search the web for FastAPI authentication' -> 'FastAPI authentication'."""
    res = resolve_web_search_arguments("Search the web for FastAPI authentication")
    assert res["query"] == "FastAPI authentication"


def test_resolver_strips_polite_opener_with_latest_qualifier():
    """2. 'Can you search the web for the latest Python release?' -> 'latest Python release'."""
    res = resolve_web_search_arguments("Can you search the web for the latest Python release?")
    assert res["query"] == "latest Python release"


def test_resolver_strips_please_find_while_preserving_recent():
    """3. 'Please find recent research on agentic AI' -> 'recent research on agentic AI'."""
    res = resolve_web_search_arguments("Please find recent research on agentic AI")
    assert res["query"] == "recent research on agentic AI"


def test_resolver_strips_search_online_for():
    """4. 'Search online for FastAPI JWT authentication' -> 'FastAPI JWT authentication'."""
    res = resolve_web_search_arguments("Search online for FastAPI JWT authentication")
    assert res["query"] == "FastAPI JWT authentication"


def test_resolver_preserves_platform_qualifiers():
    """5. 'Could you look up the latest Python release for Windows ARM64?' -> 'latest Python release for Windows ARM64'."""
    res = resolve_web_search_arguments("Could you look up the latest Python release for Windows ARM64?")
    assert res["query"] == "latest Python release for Windows ARM64"


def test_resolver_preserves_urls():
    """6. 'Search the web for https://fastapi.tiangolo.com security' retains the exact URL."""
    res = resolve_web_search_arguments("Search the web for https://fastapi.tiangolo.com security")
    assert res["query"] == "https://fastapi.tiangolo.com security"
    assert "https://fastapi.tiangolo.com" in res["query"]


@pytest.mark.parametrize(
    "clean_query",
    [
        "latest AI news",
        "FastAPI authentication",
        "Python 3.14 release",
    ],
)
def test_resolver_leaves_already_clean_query_unchanged(clean_query):
    """7. Already clean queries remain unchanged."""
    res = resolve_web_search_arguments(clean_query)
    assert res["query"] == clean_query


def test_resolver_preserves_all_qualifiers():
    """8. Preserves official, version numbers, dates, locations, and hardware targets."""
    res1 = resolve_web_search_arguments("Look up 2026 AI hardware benchmarks for Nvidia Blackwell")
    assert res1["query"] == "2026 AI hardware benchmarks for Nvidia Blackwell"

    res2 = resolve_web_search_arguments("Can you find official documentation for PostgreSQL 16 in India?")
    assert res2["query"] == "official documentation for PostgreSQL 16 in India"


def test_existing_web_search_routing_uses_clean_query():
    """9. Policy routes natural-language search requests with clean queries."""
    routed = apply_policy(
        "Can you search the web for the latest Python release for Windows ARM64?",
        _base_decision(),
    )
    assert routed.route == "tool"
    assert routed.tools == ["web_search"]
    assert routed.tool_arguments["query"] == "latest Python release for Windows ARM64"


def test_existing_memory_and_datetime_routing_unaffected():
    """10. Existing Memory and DateTime routing remain strictly unaffected."""
    # Memory profile query
    routed_mem = apply_policy("What do you know about me?", _base_decision())
    assert routed_mem.route == "memory"
    assert routed_mem.tools == []

    # Memory specific question
    decision_name = _base_decision()
    decision_name.tools = ["remember_memory"]
    decision_name.route = "tool"
    routed_name = apply_policy("What is my name?", decision_name)
    assert routed_name.route == "memory"
    assert routed_name.tools == []

    # DateTime time query
    routed_time = apply_policy("What time is it?", _base_decision())
    assert routed_time.route == "tool"
    assert routed_time.tools == ["datetime"]
    assert routed_time.tool_arguments["query_type"] == "time"

    # DateTime date query
    routed_date = apply_policy("What's today's date?", _base_decision())
    assert routed_date.route == "tool"
    assert routed_date.tools == ["datetime"]
    assert routed_date.tool_arguments["query_type"] == "date"


# =====================================================================
# P3 — FRESHNESS, DOMAIN, AND RESULT COUNT RESOLVER TESTS
# =====================================================================

def test_resolver_extracts_latest_and_recent_freshness():
    """Verify 'latest' and 'recent' freshness extraction, and that they do not silently collapse to past_week."""
    res1 = resolve_web_search_arguments("Search the web for the latest Python release")
    assert res1["freshness"] == "latest"
    assert res1["freshness"] != "past_week"
    assert "latest Python release" in res1["query"]

    res2 = resolve_web_search_arguments("Find recent research on agentic AI")
    assert res2["freshness"] == "latest"
    assert res2["freshness"] != "past_week"
    assert "recent research on agentic AI" in res2["query"]

    res3 = resolve_web_search_arguments("Search the web for newest developments in LLMs")
    assert res3["freshness"] == "latest"
    assert res3["freshness"] != "past_week"
    assert "newest developments in LLMs" in res3["query"]

    res4 = resolve_web_search_arguments("Look up breaking news about space exploration")
    assert res4["freshness"] == "latest"
    assert res4["freshness"] != "past_week"
    assert "breaking news about space exploration" in res4["query"]


def test_resolver_extracts_today_and_yesterday_freshness():
    """Verify 'today', 'past 24 hours', and 'yesterday' extraction, preserving distinct semantics."""
    res_today = resolve_web_search_arguments("Search for AI news today")
    assert res_today["freshness"] == "past_24_hours"

    res_24h = resolve_web_search_arguments("Search for stock updates in the past 24 hours")
    assert res_24h["freshness"] == "past_24_hours"

    res_yest = resolve_web_search_arguments("Search the web for what happened yesterday in AI")
    assert res_yest["freshness"] == "yesterday"
    assert res_yest["freshness"] != "past_24_hours"
    assert "yesterday in AI" in res_yest["query"]


def test_resolver_extracts_last_week_and_month_freshness():
    """Verify 'last week' and 'last month' extraction."""
    res_week = resolve_web_search_arguments("Search for Python 3.14 news from the last week")
    assert res_week["freshness"] == "past_week"
    assert res_week["query"] == "Python 3.14 news"

    res_month = resolve_web_search_arguments("Find AI breakthroughs in the past month")
    assert res_month["freshness"] == "past_month"

    res_year = resolve_web_search_arguments("Find tech breakthroughs in the past year")
    assert res_year["freshness"] == "past_year"


def test_resolver_preserves_explicit_year():
    """Verify explicit years (e.g. 2026) are preserved in the query."""
    res = resolve_web_search_arguments("Search for AI hardware benchmarks in 2026")
    assert "2026" in res["query"]


def test_resolver_extracts_explicit_domain():
    """Verify domain extraction from 'Search github.com for...'."""
    res = resolve_web_search_arguments("Search github.com for FastAPI OAuth2 examples")
    assert res["domain"] == "github.com"
    assert "FastAPI OAuth2 examples" in res["query"]
    assert "github.com" not in res["query"]


def test_resolver_extracts_site_domain():
    """Verify domain extraction from 'site:docs.python.org'."""
    res = resolve_web_search_arguments("site:docs.python.org Python tutorial")
    assert res["domain"] == "docs.python.org"
    assert res["query"] == "Python tutorial"


def test_resolver_rejects_malformed_domains():
    """Verify IP addresses, localhost, and malformed strings are not accepted as valid domains."""
    res_local = resolve_web_search_arguments("site:localhost secret admin")
    assert res_local["domain"] is None

    res_ip = resolve_web_search_arguments("site:127.0.0.1 internal status")
    assert res_ip["domain"] is None

    res_bad = resolve_web_search_arguments("site:invalid..domain query")
    assert res_bad["domain"] is None


def test_resolver_extracts_result_count():
    """Verify extraction and bounding of max_results."""
    res_3 = resolve_web_search_arguments("Search for 3 web results on Python 3.14")
    assert res_3["max_results"] == 3

    res_top8 = resolve_web_search_arguments("Search for top 8 results on FastAPI")
    assert res_top8["max_results"] == 8

    # Bounded to at most 10
    res_excess = resolve_web_search_arguments("Find limit to 50 results on AI")
    assert res_excess["max_results"] <= 10


def test_resolver_maps_official_documentation_domains():
    """Verify deterministic mapping for official documentation requests."""
    res_fastapi = resolve_web_search_arguments("Search official FastAPI documentation")
    assert res_fastapi["domain"] == "fastapi.tiangolo.com"

    res_py = resolve_web_search_arguments("Search official Python docs for asyncio")
    assert res_py["domain"] == "docs.python.org"

