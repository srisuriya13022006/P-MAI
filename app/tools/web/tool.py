import os
import re
from html import unescape
from typing import Any

import requests
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional backend dependency
    TavilyClient = None

from app.core.config import settings
from app.tools.base import BaseTool
from app.tools.result import ToolResult
from app.tools.web.url import is_valid_domain

TAVILY_TIME_RANGE_MAP = {
    "past_24_hours": "day",
    "day": "day",
    "past_week": "week",
    "week": "week",
    "past_month": "month",
    "month": "month",
    "past_year": "year",
    "year": "year",
}


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, description="The information to search for on the web.")
    freshness: str | None = Field(default=None, description="Optional freshness constraint (e.g., latest, past_24_hours, past_week, past_month, past_year).")
    domain: str | None = Field(default=None, description="Optional domain restriction (e.g., 'github.com').")
    max_results: int | None = Field(default=5, ge=1, le=10, description="Maximum number of search results to return.")
    candidate_pool_size: int | None = Field(default=None, ge=1, le=10, description="Optional internal candidate pool limit for provider retrieval.")


class WebSearchTool(BaseTool):
    """Search the web for current information using a lightweight HTML search endpoint."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current, recent, or latest information."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WebSearchInput

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute a web search using Tavily as the primary API provider, falling back to

        best-effort DuckDuckGo HTML scraping if Tavily is unavailable or encounters an error.
        """
        query = ""
        try:
            arguments = self.input_schema.model_validate(kwargs)
            query = arguments.query
            visible_max_results = min(max(1, arguments.max_results or getattr(settings, "web_search_max_results", 5)), 10)
            
            # Determine provider candidate fetch count:
            # If candidate_pool_size was explicitly requested, use it bounded [visible_max_results, 10].
            # If default max_results used, safe candidate pool from settings is applied.
            candidate_pool = arguments.candidate_pool_size or getattr(settings, "web_search_candidate_pool_size", None)
            if arguments.candidate_pool_size is not None:
                provider_max_results = min(max(visible_max_results, arguments.candidate_pool_size), 10)
            elif candidate_pool and arguments.max_results is None:
                provider_max_results = min(max(visible_max_results, candidate_pool), 10)
            else:
                provider_max_results = visible_max_results

            domain = arguments.domain if (arguments.domain and is_valid_domain(arguments.domain)) else None
            freshness = arguments.freshness
            time_range = TAVILY_TIME_RANGE_MAP.get(freshness.lower().strip()) if freshness else None

            tavily_api_key = settings.tavily_api_key or os.environ.get("TAVILY_API_KEY")

            # 1. Primary Provider: Tavily API
            if tavily_api_key and TavilyClient is not None:
                try:
                    response = self._run_tavily_search(
                        query=arguments.query,
                        api_key=tavily_api_key,
                        max_results=provider_max_results,
                        time_range=time_range,
                        domain=domain,
                    )
                    if response and response.get("results"):
                        return ToolResult(
                            tool_name=self.name,
                            success=True,
                            data={
                                "query": arguments.query,
                                "results": response["results"],
                                "answer": response.get("answer"),
                                "freshness": freshness,
                                "domain": domain,
                                "max_results": visible_max_results,
                                "candidate_pool_size": provider_max_results,
                            },
                        )
                except Exception:
                    # Fall through to secondary provider on timeout or API error
                    pass

            # 2. Secondary Provider: Best-effort HTML scraping (unreliable due to bot challenges)
            ddg_query = arguments.query
            if domain and f"site:{domain}" not in ddg_query.lower():
                ddg_query = f"site:{domain} {ddg_query}"

            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": ddg_query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                timeout=6,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()

            if getattr(response, "status_code", 200) not in (200, 202):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={"results": [], "query": arguments.query, "max_results": visible_max_results},
                    error=f"Search provider returned status {getattr(response, 'status_code', 'unknown')}.",
                )

            results = self._extract_results(response.text)
            if not results:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    data={"results": [], "query": arguments.query, "max_results": visible_max_results},
                    error="No web results were returned for the query.",
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": arguments.query,
                    "results": results[:provider_max_results],
                    "freshness": freshness,
                    "domain": domain,
                    "max_results": visible_max_results,
                    "candidate_pool_size": provider_max_results,
                },
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={"results": [], "query": query},
                error=str(exc),
            )

    def _run_tavily_search(
        self,
        query: str,
        api_key: str,
        max_results: int = 5,
        time_range: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        client = TavilyClient(api_key=api_key)
        kwargs: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        }
        if time_range:
            kwargs["time_range"] = time_range
        if domain:
            kwargs["include_domains"] = [domain]

        try:
            raw_response = client.search(**kwargs)
        except Exception:
            # Fall back to base search if advanced parameters fail on provider
            if "time_range" in kwargs or "include_domains" in kwargs:
                raw_response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=False,
                )
            else:
                raise

        results: list[dict[str, str]] = []
        for item in raw_response.get("results", [])[:max_results]:
            title = item.get("title") or "Untitled result"
            url = item.get("url") or ""
            snippet = item.get("content") or item.get("snippet") or ""
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        return {
            "results": results,
            "answer": raw_response.get("answer"),
        }

    def _extract_results(self, html_text: str) -> list[dict[str, str]]:
        matches = re.findall(
            r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        snippets = re.findall(
            r'<a[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</a>',
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: list[dict[str, str]] = []
        seen_links: set[str] = set()
        for index, (link, title) in enumerate(matches[:10]):
            normalized_link = self._normalize_url(link)
            if not normalized_link or normalized_link in seen_links:
                continue
            clean_title = self._strip_tags(title)
            if not clean_title or self._is_generic_search_page(normalized_link, clean_title):
                continue
            snippet = self._strip_tags(snippets[index]) if index < len(snippets) else ""
            results.append(
                {
                    "title": clean_title,
                    "url": normalized_link,
                    "snippet": snippet,
                }
            )
            seen_links.add(normalized_link)

        if len(results) < 2:
            fallback = re.findall(
                r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                html_text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for link, title in fallback[:10]:
                normalized_link = self._normalize_url(link)
                if not normalized_link or normalized_link in seen_links:
                    continue
                clean_title = self._strip_tags(title)
                if not clean_title or self._is_generic_search_page(normalized_link, clean_title):
                    continue
                results.append(
                    {
                        "title": clean_title,
                        "url": normalized_link,
                        "snippet": "",
                    }
                )
                seen_links.add(normalized_link)

        return results[:5]

    @staticmethod
    def _normalize_url(url: str) -> str:
        text = url.strip()
        if not text or text.startswith("javascript:"):
            return ""
        if text.startswith("//"):
            text = "https:" + text
        if text.startswith("/"):
            text = f"https://duckduckgo.com{text}"
        return text

    @staticmethod
    def _is_generic_search_page(url: str, title: str) -> bool:
        lower_url = url.lower()
        lower_title = title.lower()
        if "duckduckgo" in lower_url or "duckduckgo" in lower_title:
            return True
        if lower_url.endswith("/html/") or lower_url.endswith("/html"):
            return True
        if lower_title in {"duckduckgo", "search", "html"}:
            return True
        return False

    @staticmethod
    def _strip_tags(value: str) -> str:
        cleaned = re.sub(r"<.*?>", " ", value, flags=re.DOTALL)
        cleaned = unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
