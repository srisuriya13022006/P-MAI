from app.agent.policy import apply_policy
from app.schemas.agent import AgentDecision
from app.tools.web.tool import WebSearchTool


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


def test_web_search_policy_routes_current_information_requests():
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )

    routed = apply_policy(
        "What are the latest AI news updates today?",
        decision.model_copy(),
    )

    assert routed.route == "tool"
    assert routed.tools == ["web_search"]
    assert routed.tool_arguments["query"].startswith("latest AI news") or "AI news" in routed.tool_arguments["query"]


def test_web_search_policy_routes_public_profile_queries():
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )

    routed = apply_policy(
        "Search the web for Srisuriya from Sri Eshwar College AI & DS student.",
        decision.model_copy(),
    )

    assert routed.route == "tool"
    assert routed.tools == ["web_search"]
    assert "Srisuriya" in routed.tool_arguments["query"]


def test_web_search_tool_ignores_generic_search_engine_pages():
    html = """
    <html>
      <body>
        <a class="result-link" href="/html/">DuckDuckGo</a>
        <a class="result-link" href="https://example.com/ai">AI breakthrough announced</a>
        <a class="result-snippet">A major model update changed the market.</a>
      </body>
    </html>
    """

    tool = WebSearchTool()
    results = tool._extract_results(html)

    assert results
    assert all("duckduckgo" not in result["title"].lower() for result in results)
    assert "AI breakthrough announced" in results[0]["title"]


def test_web_search_tool_uses_tavily_when_available(monkeypatch):
    class FakeTavilyClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key

        def search(self, query, **kwargs):
            return {
                "results": [
                    {"title": "AI breakthrough announced", "url": "https://example.com/ai", "content": "A major model update changed the market."},
                    {"title": "Future of AI", "url": "https://example.com/future", "content": "Experts discuss the next wave of foundation models."},
                ],
                "answer": "AI is changing quickly.",
            }

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", FakeTavilyClient)

    tool = WebSearchTool()
    result = tool.run(query="AI breakthroughs")

    assert result.success is True
    assert len(result.data["results"]) >= 2
    assert "AI breakthrough announced" in result.data["results"][0]["title"]
    assert "AI is changing quickly." in result.data.get("answer", "")


def test_web_search_tool_returns_results_from_search_html(monkeypatch):
    html = """
    <html>
      <body>
        <a class="result-link" href="https://example.com/ai">AI breakthrough announced</a>
        <a class="result-snippet">A major model update changed the market.</a>
        <a class="result-link" href="https://example.com/future">Future of AI</a>
        <a class="result-snippet">Experts discuss the next wave of foundation models.</a>
      </body>
    </html>
    """

    def fake_get(url, params=None, headers=None, timeout=None):
        return FakeResponse(html)

    monkeypatch.setattr("app.tools.web.tool.requests.get", fake_get)
    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", None)

    tool = WebSearchTool()
    result = tool.run(query="AI breakthroughs")

    assert result.success is True
    assert len(result.data["results"]) >= 2
    assert "AI breakthrough announced" in result.data["results"][0]["title"]


import pytest
from app.agent.orchestrator import MAIOrchestrator


@pytest.mark.parametrize(
    "query",
    [
        "Tell me about quantum computing.",
        "Who is Albert Einstein?",
        "What is artificial intelligence?",
        "What engineering branches exist?",
    ],
)
def test_routing_negative_static_knowledge_queries(query):
    """Static knowledge queries must not be force-routed to web_search."""
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )
    routed = apply_policy(query, decision.model_copy())
    assert routed.route != "tool" or "web_search" not in routed.tools


@pytest.mark.parametrize(
    "query",
    [
        "What is the latest AI news?",
        "What are the recent OpenAI updates?",
        "Search the web for FastAPI authentication.",
        "Find recent research on agentic AI.",
        "What happened today in AI?",
    ],
)
def test_routing_positive_current_and_explicit_queries(query):
    """Current information and explicit search queries must route to web_search."""
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )
    routed = apply_policy(query, decision.model_copy())
    assert routed.route == "tool"
    assert routed.tools == ["web_search"]
    assert "query" in routed.tool_arguments


def test_routing_preserves_memory_and_datetime_paths():
    """Confirms memory and datetime paths remain unaffected by web search policy."""
    decision = AgentDecision(
        intent="local",
        route="local",
        needs_clarification=False,
        tools=[],
        reason="test",
    )
    routed_mem = apply_policy("What do you know about me?", decision.model_copy())
    assert routed_mem.route == "memory"
    assert "web_search" not in routed_mem.tools

    routed_dt = apply_policy("What time is it?", decision.model_copy())
    assert routed_dt.route == "tool"
    assert routed_dt.tools == ["datetime"]


def test_orchestrator_web_search_uses_local_llm_when_cloud_unavailable():
    """When cloud LLM is unavailable, local LLM synthesizes web search results."""
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
                tool_arguments={"query": "latest python release"},
                reason="web search",
            )
        },
    )()

    class FakeWebTool:
        name = "web_search"

        def run(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": {
                        "query": kwargs["query"],
                        "results": [
                            {
                                "title": "Python 3.14 Released",
                                "url": "https://python.org/release/3.14",
                                "snippet": "Python 3.14 is now available with new features.",
                            }
                        ],
                    },
                    "error": None,
                    "tool_name": "web_search",
                },
            )()

    calls = []

    class FakeLocalLLM:
        def generate(self, messages, system_prompt=None):
            calls.append(messages[0]["content"])
            return "Python 3.14 was recently released. [Python 3.14 Released](https://python.org/release/3.14)"

    orchestrator.tool_registry._tools["web_search"] = FakeWebTool()
    orchestrator.cloud_llm = None  # Cloud LLM is unavailable
    orchestrator.llm = FakeLocalLLM()

    response = orchestrator.handle("What is the latest python release?", [])

    # Local LLM was invoked
    assert len(calls) == 1
    # Synthesized answer returned, not raw snippets
    assert "Python 3.14 was recently released." in response
    assert "https://python.org/release/3.14" in response
    assert "SOURCE 1:" not in response


def test_orchestrator_web_search_prompt_contains_security_and_citation_guidelines():
    """Verifies that synthesis prompt contains security boundaries and citation contracts."""
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
                tool_arguments={"query": "FastAPI security"},
                reason="web search",
            )
        },
    )()

    class FakeWebTool:
        name = "web_search"

        def run(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": {
                        "query": kwargs["query"],
                        "results": [
                            {
                                "title": "FastAPI Security Docs",
                                "url": "https://fastapi.tiangolo.com/tutorial/security/",
                                "snippet": "FastAPI provides tools for security and authentication.",
                            }
                        ],
                    },
                    "error": None,
                    "tool_name": "web_search",
                },
            )()

    captured_prompt = []

    class FakeLocalLLM:
        def generate(self, messages, system_prompt=None):
            captured_prompt.append(messages[0]["content"])
            return "FastAPI includes security utilities. [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)"

    orchestrator.tool_registry._tools["web_search"] = FakeWebTool()
    orchestrator.cloud_llm = None
    orchestrator.llm = FakeLocalLLM()

    orchestrator.handle("How does FastAPI handle security?", [])

    assert len(captured_prompt) == 1
    prompt_text = captured_prompt[0]
    assert "<search_evidence>" in prompt_text
    assert "</search_evidence>" in prompt_text
    assert "untrusted external data" in prompt_text
    assert "cite the source using Markdown links" in prompt_text
    assert "https://fastapi.tiangolo.com/tutorial/security/" in prompt_text


def test_orchestrator_handles_synthesis_failure_gracefully():
    """When both cloud and local LLMs fail during synthesis, clean error is returned."""
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
                tool_arguments={"query": "test query"},
                reason="web search",
            )
        },
    )()

    class FakeWebTool:
        name = "web_search"

        def run(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": {
                        "query": kwargs["query"],
                        "results": [
                            {"title": "Result", "url": "https://example.com", "snippet": "Snippet"}
                        ],
                    },
                    "error": None,
                    "tool_name": "web_search",
                },
            )()

    class BrokenLLM:
        def generate(self, messages, system_prompt=None):
            raise RuntimeError("LLM synthesis model crashed")

    orchestrator.tool_registry._tools["web_search"] = FakeWebTool()
    orchestrator.cloud_llm = BrokenLLM()
    orchestrator.llm = BrokenLLM()

    response = orchestrator.handle("Search for test query", [])
    assert "error while synthesizing the answer" in response


def test_web_search_tool_handles_tavily_failure_falling_back_to_html(monkeypatch):
    """When Tavily throws an exception, WebSearchTool falls back to HTML scraping."""
    html = """
    <html>
      <body>
        <a class="result-link" href="https://example.com/fallback">Fallback Result</a>
        <a class="result-snippet">Recovered from fallback search.</a>
      </body>
    </html>
    """

    class BrokenTavilyClient:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            raise TimeoutError("Tavily API timed out")

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "valid-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", BrokenTavilyClient)
    monkeypatch.setattr(
        "app.tools.web.tool.requests.get",
        lambda url, params=None, headers=None, timeout=None: FakeResponse(html),
    )

    tool = WebSearchTool()
    result = tool.run(query="fallback test")

    assert result.success is True
    assert len(result.data["results"]) >= 1
    assert "Fallback Result" in result.data["results"][0]["title"]


def test_web_search_tool_handles_both_providers_failing_gracefully(monkeypatch):
    """When both Tavily and HTML scraping fail, returns clean failure without crash."""
    class BrokenTavilyClient:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            raise TimeoutError("Tavily API timed out")

    def broken_get(url, params=None, headers=None, timeout=None):
        raise ConnectionError("Connection refused by fallback")

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "valid-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", BrokenTavilyClient)
    monkeypatch.setattr("app.tools.web.tool.requests.get", broken_get)

    tool = WebSearchTool()
    result = tool.run(query="failing test")

    assert result.success is False
    assert result.error is not None
    assert "results" in result.data
    assert len(result.data["results"]) == 0


# =====================================================================
# P3 — QUALITY, FRESHNESS, DOMAIN, AND FALLBACK TESTS
# =====================================================================

def test_web_search_tavily_freshness_and_domain_mapping(monkeypatch):
    """Verify freshness and domain filters are passed to Tavily search client."""
    captured_kwargs = {}

    class MockTavilyClient:
        def __init__(self, api_key=None, **kwargs):
            pass

        def search(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return {
                "results": [
                    {"title": "FastAPI Examples", "url": "https://github.com/fastapi/fastapi", "content": "FastAPI code on GitHub"},
                ],
                "answer": "Here are GitHub examples.",
            }

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", MockTavilyClient)

    tool = WebSearchTool()
    result = tool.run(
        query="FastAPI OAuth2 examples",
        freshness="past_week",
        domain="github.com",
        max_results=3,
    )

    assert result.success is True
    assert captured_kwargs["query"] == "FastAPI OAuth2 examples"
    assert captured_kwargs["time_range"] == "week"
    assert captured_kwargs["include_domains"] == ["github.com"]
    assert captured_kwargs["max_results"] == 3
    assert result.data["domain"] == "github.com"
    assert result.data["freshness"] == "past_week"


@pytest.mark.parametrize(
    "freshness_val,expected_time_range",
    [
        ("past_24_hours", "day"),
        ("past_week", "week"),
        ("past_month", "month"),
        ("past_year", "year"),
        ("latest", None),
        ("recent", None),
        ("newest", None),
        ("breaking news", None),
        ("yesterday", None),
        ("unsupported_custom_range", None),
    ],
)
def test_web_search_tavily_freshness_semantics_and_mapping(monkeypatch, freshness_val, expected_time_range):
    """Verify that only exact interval mappings (day, week, month, year) set Tavily time_range,
    while 'latest', 'recent', 'newest', 'breaking news', and 'yesterday' do NOT collapse
    into arbitrary provider filters and instead rely safely on semantic query qualifiers."""
    captured_kwargs = {}

    class MockTavilyClient:
        def __init__(self, api_key=None, **kwargs):
            pass

        def search(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return {
                "results": [
                    {"title": "Result", "url": "https://example.com/item", "content": "Content"},
                ],
            }

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", MockTavilyClient)

    tool = WebSearchTool()
    result = tool.run(
        query="Test query",
        freshness=freshness_val,
    )

    assert result.success is True
    assert result.data["freshness"] == freshness_val
    if expected_time_range is not None:
        assert captured_kwargs.get("time_range") == expected_time_range
    else:
        assert "time_range" not in captured_kwargs, f"Expected no time_range for {freshness_val}, got {captured_kwargs.get('time_range')}"


def test_web_search_duckduckgo_fallback_preserves_freshness_metadata(monkeypatch):
    """Verify DuckDuckGo fallback preserves internal freshness metadata and does not claim false filtering."""
    captured_params = {}

    html = """
    <html><body>
      <a class="result-link" href="https://example.com/article">Article Title</a>
      <a class="result-snippet">Article snippet content</a>
    </body></html>
    """

    def mock_get(url, params=None, headers=None, timeout=None):
        nonlocal captured_params
        captured_params = params
        return FakeResponse(html)

    # Disable Tavily to force DuckDuckGo fallback
    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", None)
    monkeypatch.setattr("app.tools.web.tool.requests.get", mock_get)

    tool = WebSearchTool()
    result = tool.run(query="what happened yesterday in AI", freshness="yesterday")

    assert result.success is True
    assert result.data["freshness"] == "yesterday"
    assert result.data["query"] == "what happened yesterday in AI"
    assert "yesterday" in captured_params["q"]
    assert len(result.data["results"]) == 1


def test_web_search_provider_fallback_when_freshness_unsupported(monkeypatch):
    """Verify that if Tavily rejects advanced filter parameters, it gracefully retries without them."""
    call_count = 0

    class StrictTavilyClient:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if "time_range" in kwargs:
                raise ValueError("Unsupported parameter time_range")
            return {
                "results": [
                    {"title": "Standard Result", "url": "https://example.com/standard", "content": "Standard search result content"},
                ],
            }

    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("app.tools.web.tool.TavilyClient", StrictTavilyClient)

    tool = WebSearchTool()
    result = tool.run(query="Python 3.14 news", freshness="past_week")

    assert result.success is True
    assert call_count == 2  # First with time_range, second without
    assert result.data["results"][0]["title"] == "Standard Result"


def test_web_search_duckduckgo_fallback_applies_domain_and_max_results(monkeypatch):
    """Verify DuckDuckGo fallback incorporates domain restriction and respects max_results."""
    captured_params = {}

    html = """
    <html><body>
      <a class="result-link" href="https://github.com/example/1">Result 1</a>
      <a class="result-snippet">Snippet 1</a>
      <a class="result-link" href="https://github.com/example/2">Result 2</a>
      <a class="result-snippet">Snippet 2</a>
      <a class="result-link" href="https://github.com/example/3">Result 3</a>
      <a class="result-snippet">Snippet 3</a>
      <a class="result-link" href="https://github.com/example/4">Result 4</a>
      <a class="result-snippet">Snippet 4</a>
    </body></html>
    """

    def mock_get(url, params=None, headers=None, timeout=None):
        nonlocal captured_params
        captured_params = params
        return FakeResponse(html)

    # Disable Tavily to force DuckDuckGo fallback
    monkeypatch.setattr("app.tools.web.tool.settings.tavily_api_key", None)
    monkeypatch.setattr("app.tools.web.tool.requests.get", mock_get)

    tool = WebSearchTool()
    result = tool.run(query="OAuth2 code", domain="github.com", max_results=2)

    assert result.success is True
    assert "site:github.com OAuth2 code" in captured_params["q"]
    assert len(result.data["results"]) == 2  # Sliced to max_results


def test_routing_separation_search_vs_web_fetch():
    """Verify strict routing boundary: explicit read requests go to web_fetch, search goes to web_search."""
    decision = AgentDecision(
        intent="general",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )

    # Search with domain -> web_search
    routed_search = apply_policy("Search github.com for FastAPI OAuth2 examples", decision.model_copy())
    assert routed_search.route == "tool"
    assert routed_search.tools == ["web_search"]
    assert routed_search.tool_arguments.get("domain") == "github.com"

    # Direct URL read -> web_fetch
    routed_fetch = apply_policy("Read https://fastapi.tiangolo.com/security and explain OAuth2", decision.model_copy())
    assert routed_fetch.route == "tool"
    assert routed_fetch.tools == ["web_fetch"]


def test_local_routing_preserved_for_non_current_queries():
    """Verify general knowledge queries without current info or explicit search intent remain local."""
    base = AgentDecision(
        intent="general",
        route="local",
        needs_clarification=False,
        tools=[],
        tool_arguments={},
        reason="test",
    )

    for msg in (
        "What is quantum computing?",
        "Who was Albert Einstein?",
        "Explain transformers",
        "What is FastAPI?",
        "What is SQL normalization?",
    ):
        routed = apply_policy(msg, base.model_copy())
        assert routed.route == "local", f"Expected local for '{msg}', got {routed.route} (tools={routed.tools})"
        assert routed.tools == []


