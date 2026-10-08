import pytest
from app.agent.orchestrator import MAIOrchestrator
from app.agent.policy import apply_policy
from app.schemas.agent import AgentDecision
from app.tools.web.url import (
    canonicalize_url,
    deduplicate_search_results,
    is_valid_web_url,
)


class MockWebTool:
    def __init__(self, results=None, answer=None, success=True, error=None):
        self.name = "web_search"
        self._results = results if results is not None else []
        self._answer = answer
        self._success = success
        self._error = error

    def run(self, **kwargs):
        return type(
            "ToolResult",
            (),
            {
                "tool_name": self.name,
                "success": self._success,
                "data": {
                    "query": kwargs.get("query"),
                    "results": self._results,
                    "answer": self._answer,
                },
                "error": self._error,
            },
        )()


class CapturingLLM:
    def __init__(self, response_text="Synthesized answer citing [FastAPI](https://example.com/page?utm_source=twitter)"):
        self.captured_prompts = []
        self.response_text = response_text

    def generate(self, messages, system_prompt=None):
        content = messages[-1]["content"] if messages else ""
        self.captured_prompts.append(content)
        return self.response_text


def _build_orchestrator(tool, llm, cloud_llm=None):
    orchestrator = MAIOrchestrator()
    orchestrator.router = type(
        "Router",
        (),
        {
            "analyze": lambda self, message: AgentDecision(
                intent="web_search",
                route="tool",
                needs_clarification=False,
                tools=["web_search"],
                tool_arguments={"query": message},
                reason="web search test",
            )
        },
    )()
    orchestrator.tool_registry._tools["web_search"] = tool
    orchestrator.llm = llm
    orchestrator.cloud_llm = cloud_llm
    return orchestrator


def _extract_evidence(prompt: str) -> str:
    start = prompt.find("<search_evidence>")
    end = prompt.find("</search_evidence>")
    if start != -1 and end != -1:
        return prompt[start + len("<search_evidence>"):end]
    return ""


# ==========================================
# 1. CANONICALIZATION UNIT TESTS
# ==========================================

def test_canonicalize_exact_duplicates():
    url1 = "https://example.com/page"
    url2 = "https://example.com/page"
    assert canonicalize_url(url1) == canonicalize_url(url2) == "https://example.com/page"


def test_canonicalize_trailing_slash():
    url1 = "https://example.com/page"
    url2 = "https://example.com/page/"
    assert canonicalize_url(url1) == canonicalize_url(url2) == "https://example.com/page"

    # Root trailing slash
    assert canonicalize_url("https://example.com/") == "https://example.com"
    assert canonicalize_url("https://example.com") == "https://example.com"


def test_canonicalize_host_and_scheme_casing():
    url1 = "https://EXAMPLE.COM/page"
    url2 = "HTTPS://example.com/page"
    url3 = "https://example.com/PAGE"  # path casing preserved
    assert canonicalize_url(url1) == "https://example.com/page"
    assert canonicalize_url(url2) == "https://example.com/page"
    assert canonicalize_url(url3) == "https://example.com/PAGE"


def test_canonicalize_strips_tracking_query_parameters():
    url1 = "https://example.com/page?utm_source=twitter&utm_medium=social"
    url2 = "https://example.com/page?gclid=12345&fbclid=abcdef"
    url3 = "https://example.com/page?utm_campaign=winter_sale&utm_content=logolink"
    base = "https://example.com/page"

    assert canonicalize_url(url1) == base
    assert canonicalize_url(url2) == base
    assert canonicalize_url(url3) == base


def test_canonicalize_strips_fragments():
    url1 = "https://example.com/page#section1"
    url2 = "https://example.com/page#section2"
    assert canonicalize_url(url1) == canonicalize_url(url2) == "https://example.com/page"


def test_canonicalize_preserves_meaningful_query_parameters():
    url1 = "https://example.com/page?page=1"
    url2 = "https://example.com/page?page=2"
    url_version1 = "https://example.com/doc?v=1"
    url_version2 = "https://example.com/doc?v=2"

    assert canonicalize_url(url1) != canonicalize_url(url2)
    assert canonicalize_url(url1) == "https://example.com/page?page=1"
    assert canonicalize_url(url2) == "https://example.com/page?page=2"

    assert canonicalize_url(url_version1) != canonicalize_url(url_version2)
    assert canonicalize_url(url_version1) == "https://example.com/doc?v=1"
    assert canonicalize_url(url_version2) == "https://example.com/doc?v=2"


def test_canonicalize_meaningful_param_with_tracking_param():
    url = "https://example.com/search?q=fastapi&utm_source=google&page=2"
    # utm_source stripped, q and page retained in sorted order
    expected = "https://example.com/search?page=2&q=fastapi"
    assert canonicalize_url(url) == expected


def test_canonicalize_rejects_unsafe_and_invalid_urls():
    assert canonicalize_url("javascript:alert(1)") == ""
    assert canonicalize_url("ftp://example.com/file.zip") == ""
    assert canonicalize_url("file:///etc/passwd") == ""
    assert canonicalize_url("data:text/html,<html>") == ""
    assert canonicalize_url("") == ""
    assert canonicalize_url("   ") == ""
    assert canonicalize_url(None) == ""


def test_is_valid_web_url_helper():
    assert is_valid_web_url("https://fastapi.tiangolo.com") is True
    assert is_valid_web_url("http://localhost:8000/docs") is True
    assert is_valid_web_url("javascript:void(0)") is False
    assert is_valid_web_url("ftp://ftp.is.co.za") is False
    assert is_valid_web_url("") is False


# ==========================================
# 2. DEDUPLICATION LOGIC TESTS
# ==========================================

def test_deduplicate_collapses_all_variant_forms():
    raw_results = [
        {"title": "Doc Clean", "url": "https://example.com/page", "snippet": "Clean snippet"},
        {"title": "Doc Trailing Slash", "url": "https://example.com/page/", "snippet": "Trailing slash snippet"},
        {"title": "Doc Host Casing", "url": "https://EXAMPLE.COM/page", "snippet": "Host casing snippet"},
        {"title": "Doc Tracker", "url": "https://example.com/page?utm_source=twitter", "snippet": "Tracker snippet"},
        {"title": "Doc Fragment", "url": "https://example.com/page#overview", "snippet": "Fragment snippet"},
    ]

    deduped = deduplicate_search_results(raw_results, max_results=5)
    assert len(deduped) == 1
    assert deduped[0]["canonical_url"] == "https://example.com/page"
    # Preserves the first encountered original_url
    assert deduped[0]["url"] == "https://example.com/page"


def test_deduplicate_preserves_original_url_for_citations():
    raw_results = [
        {
            "title": "FastAPI OAuth2",
            "url": "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/?utm_source=chat",
            "snippet": "OAuth2 tutorial",
        }
    ]

    deduped = deduplicate_search_results(raw_results, max_results=5)
    assert len(deduped) == 1
    # Canonical URL is cleaned
    assert deduped[0]["canonical_url"] == "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt"
    # Original URL is retained verbatim for user citations
    assert deduped[0]["url"] == "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/?utm_source=chat"


def test_deduplicate_eight_results_with_three_duplicate_groups():
    """8 results with 3 duplicate groups -> exactly 5 unique sources."""
    raw_results = [
        # Group 1: Page A
        {"title": "Page A", "url": "https://example.com/a", "snippet": "Snippet A1"},
        {"title": "Page A Copy", "url": "https://example.com/a?utm_source=newsletter", "snippet": "Snippet A2"},
        # Group 2: Page B
        {"title": "Page B", "url": "https://example.com/b/", "snippet": "Snippet B1"},
        {"title": "Page B Fragment", "url": "https://example.com/b#faq", "snippet": "Snippet B2"},
        # Group 3: Page C
        {"title": "Page C", "url": "https://example.com/c", "snippet": "Snippet C1"},
        {"title": "Page C Cased", "url": "https://EXAMPLE.COM/c/", "snippet": "Snippet C2"},
        # Group 4: Page D
        {"title": "Page D", "url": "https://example.com/d", "snippet": "Snippet D"},
        # Group 5: Page E
        {"title": "Page E", "url": "https://example.com/e", "snippet": "Snippet E"},
    ]

    deduped = deduplicate_search_results(raw_results, max_results=5)
    assert len(deduped) == 5
    canonical_urls = [item["canonical_url"] for item in deduped]
    assert canonical_urls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
        "https://example.com/d",
        "https://example.com/e",
    ]


def test_deduplicate_ignores_invalid_and_empty_urls():
    raw_results = [
        {"title": "Empty", "url": "", "snippet": "no url"},
        {"title": "Javascript", "url": "javascript:void(0)", "snippet": "bad scheme"},
        {"title": "FTP", "url": "ftp://files.example.com", "snippet": "bad scheme"},
        {"title": "Valid", "url": "https://example.com/valid", "snippet": "good url"},
    ]

    deduped = deduplicate_search_results(raw_results, max_results=5)
    assert len(deduped) == 1
    assert deduped[0]["canonical_url"] == "https://example.com/valid"


def test_deduplicate_conservatively_upgrades_untitled_placeholder():
    raw_results = [
        {"title": "Untitled result", "url": "https://example.com/guide", "snippet": "short"},
        {"title": "Official Python Guide", "url": "https://example.com/guide/", "snippet": "A very comprehensive guide to Python."},
    ]

    deduped = deduplicate_search_results(raw_results, max_results=5)
    assert len(deduped) == 1
    assert deduped[0]["title"] == "Official Python Guide"
    assert deduped[0]["snippet"] == "A very comprehensive guide to Python."


# ==========================================
# 3. FULL ORCHESTRATOR INTEGRATION TESTS
# ==========================================

def test_orchestrator_formats_canonical_deduplicated_sources_with_original_urls():
    """Orchestrator uses canonical deduplication but preserves original provider URL in prompt."""
    raw_results = [
        {
            "title": "FastAPI Docs",
            "url": "https://fastapi.tiangolo.com/tutorial/?utm_source=google",
            "snippet": "FastAPI tutorial",
        },
        {
            "title": "FastAPI Docs Duplicate",
            "url": "https://fastapi.tiangolo.com/tutorial#intro",
            "snippet": "Duplicate intro",
        },
        {
            "title": "FastAPI Advanced",
            "url": "https://fastapi.tiangolo.com/advanced/",
            "snippet": "Advanced tutorial",
        },
    ]

    tool = MockWebTool(results=raw_results)
    llm = CapturingLLM()
    orchestrator = _build_orchestrator(tool, llm)

    response = orchestrator.handle("FastAPI tutorial", [])

    assert len(llm.captured_prompts) == 1
    evidence = _extract_evidence(llm.captured_prompts[0])

    # Exactly 2 sources in evidence (the duplicate was suppressed)
    assert "<SOURCE_1>" in evidence
    assert "<SOURCE_2>" in evidence
    assert "<SOURCE_3>" not in evidence

    # Exact original URL preserved for user citations
    assert "https://fastapi.tiangolo.com/tutorial/?utm_source=google" in evidence
    assert "https://fastapi.tiangolo.com/advanced/" in evidence


def test_orchestrator_regression_routing_and_datetime():
    """Verify policy routing for memory, datetime, and web search remains unaffected."""
    base = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )

    # Web search
    web_res = apply_policy("Search the web for FastAPI", base.model_copy())
    assert web_res.route == "tool"
    assert web_res.tools == ["web_search"]

    # DateTime
    dt_res = apply_policy("What time is it?", base.model_copy())
    assert dt_res.route == "tool"
    assert dt_res.tools == ["datetime"]

    # Memory
    mem_res = apply_policy("What do you know about me?", base.model_copy())
    assert mem_res.route == "memory"
    assert mem_res.tools == []
