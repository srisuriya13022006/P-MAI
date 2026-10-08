"""Deterministic Web Search Result Quality and Ranking Model (P4).

Scores, ranks, and filters web search results prior to synthesis.
Uses explainable, deterministic signals:
- Lexical textual relevance (title overlap, snippet overlap, exact phrase match, version numbers)
- Source / domain authority (official docs, academic research, reputable news)
- Query-aware source alignment (entity match, technical/research/news intent)
- Freshness signal (uses provider publication dates when available; neutral when missing)
- Soft domain diversity penalty (mitigates domain concentration without rigid elimination)

Does NOT use an LLM or embeddings. Operates independently of search provider.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.agent.tool_argument_resolver import OFFICIAL_DOCS_MAPPING
from app.tools.web.url import canonicalize_url

# Standard English stopwords to exclude from lexical term matching
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "too", "very", "can", "will", "just", "don", "should",
    "now", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "search", "web", "find", "look", "tell",
}

# High-authority domains for technical documentation and primary sources
PRIMARY_TECH_DOMAINS = {
    "docs.python.org",
    "python.org",
    "fastapi.tiangolo.com",
    "developer.mozilla.org",
    "docs.github.com",
    "github.com",
    "learn.microsoft.com",
    "microsoft.com",
    "aws.amazon.com",
    "cloud.google.com",
    "developers.google.com",
    "openai.com",
    "pytorch.org",
    "tensorflow.org",
    "pypi.org",
    "npmjs.com",
    "rust-lang.org",
    "go.dev",
    "golang.org",
    "docker.com",
    "docs.docker.com",
    "kubernetes.io",
    "postgresql.org",
    "sqlite.org",
    "react.dev",
    "vuejs.org",
    "angular.dev",
    "nodejs.org",
}

# Authoritative academic and scientific research publishers
ACADEMIC_DOMAINS = {
    "arxiv.org",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "biorxiv.org",
    "medrxiv.org",
    "semanticscholar.org",
    "openreview.net",
    "aclweb.org",
    "neurips.cc",
    "iclr.cc",
    "icml.cc",
}

# Reputable news publishers for current events
REPUTABLE_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "bbc.com",
    "bbc.co.uk",
    "theverge.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "venturebeat.com",
}


def extract_hostname(url: str) -> str:
    """Extract clean lowercase hostname from URL without leading 'www.'."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower().strip()
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def extract_registered_domain(hostname: str) -> str:
    """Extract registered domain (apex domain) from hostname.
    e.g. 'docs.python.org' -> 'python.org', 'fastapi.tiangolo.com' -> 'tiangolo.com'
    """
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def classify_source_type(hostname: str, url: str) -> str:
    """Deterministically classify the source type based on domain and URL path."""
    if not hostname:
        return "general_web"

    # Check official documentation domains
    if hostname in PRIMARY_TECH_DOMAINS or any(hostname.endswith("." + d) for d in PRIMARY_TECH_DOMAINS):
        return "official_docs"

    # Any domain in official mapping
    for proj_domain in OFFICIAL_DOCS_MAPPING.values():
        if hostname == proj_domain or hostname.endswith("." + proj_domain):
            return "official_docs"

    # Domain starting with docs. or URL path containing /docs/
    if hostname.startswith("docs.") or any(f"/{p}/" in url.lower() for p in ("docs", "documentation", "reference", "manual")):
        return "official_docs"

    # Academic domains
    if hostname in ACADEMIC_DOMAINS or any(hostname.endswith("." + d) for d in ACADEMIC_DOMAINS):
        return "academic_research"
    if hostname.endswith(".edu") or hostname.endswith(".ac.uk") or ".edu." in hostname:
        return "academic_research"

    # News domains
    if hostname in REPUTABLE_NEWS_DOMAINS or any(hostname.endswith("." + d) for d in REPUTABLE_NEWS_DOMAINS):
        return "authoritative_news"

    return "general_web"


def tokenize_text(text: str) -> list[str]:
    """Tokenize string into non-stopword alphanumeric tokens."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\.\-]+\b", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def compute_textual_relevance(
    query: str,
    title: str,
    snippet: str,
) -> float:
    """Calculate deterministic lexical relevance score in [0.0, 1.0].
    
    Signals:
    - Non-stopword query token coverage in title (weight: 0.40)
    - Key phrase overlap in title (weight: 0.20)
    - Token coverage in snippet (weight: 0.20)
    - Version number match / conflict in title or snippet (up to +0.20 or -0.15)
    """
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return 0.5

    title_lower = (title or "").lower()
    snippet_lower = (snippet or "").lower()

    # 1. Title token coverage
    title_matches = sum(1 for tok in query_tokens if tok in title_lower)
    title_coverage = title_matches / len(query_tokens)

    # 2. Snippet token coverage
    snippet_matches = sum(1 for tok in query_tokens if tok in snippet_lower)
    snippet_coverage = snippet_matches / len(query_tokens)

    # 3. Phrase match
    phrase_score = 0.0
    clean_query = " ".join(query_tokens)
    if len(query_tokens) >= 2 and clean_query in title_lower:
        phrase_score = 0.20
    elif len(query_tokens) >= 2 and clean_query in snippet_lower:
        phrase_score = 0.10

    # 4. Version number matching (e.g. 3.14, 2.0)
    version_adjustment = 0.0
    query_versions = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", query)
    if query_versions:
        for ver in query_versions:
            if ver in title_lower:
                version_adjustment += 0.20
                break
            elif ver in snippet_lower:
                version_adjustment += 0.10
                break
        else:
            # Result contains version numbers, but none match requested version
            result_versions = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", title_lower + " " + snippet_lower)
            if result_versions and not any(v in query_versions for v in result_versions):
                version_adjustment -= 0.15

    base_relevance = (0.40 * title_coverage) + (0.25 * snippet_coverage) + phrase_score + version_adjustment
    return max(0.0, min(1.0, base_relevance))


def compute_authority_score(source_type: str, hostname: str, url: str) -> float:
    """Calculate base source authority score in [0.0, 1.0]."""
    if source_type == "official_docs":
        score = 0.88
    elif source_type == "academic_research":
        score = 0.85
    elif source_type == "authoritative_news":
        score = 0.75
    else:
        score = 0.50

    # Clean URL bonus (well-structured docs or release path)
    url_lower = url.lower()
    if any(p in url_lower for p in ("/docs", "/release", "/tutorial", "/security", "/api", "/paper", "/abs")):
        score += 0.05

    return max(0.0, min(1.0, score))


def compute_query_intent_alignment(
    query: str,
    source_type: str,
    hostname: str,
) -> float:
    """Evaluate alignment between query intent and source characteristics in [0.0, 1.0]."""
    query_lower = query.lower()
    alignment = 0.0

    # Entity alignment: if query mentions entity and domain matches it
    # e.g. "fastapi" in query and "fastapi" in hostname
    query_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", query_lower))
    for word in query_words:
        if len(word) >= 4 and word in hostname:
            alignment += 0.25
            break

    # Technical query alignment
    is_tech = any(w in query_words for w in ("fastapi", "python", "docker", "kubernetes", "oauth2", "jwt", "api", "release", "install", "tutorial", "code"))
    if is_tech and source_type == "official_docs":
        alignment += 0.15

    # Research query alignment
    is_research = any(w in query_words for w in ("research", "paper", "papers", "study", "agentic", "benchmark", "model", "survey"))
    if is_research and source_type == "academic_research":
        alignment += 0.20

    # News query alignment
    is_news = any(w in query_words for w in ("news", "today", "yesterday", "latest", "recent", "breaking", "announced"))
    if is_news and source_type in ("authoritative_news", "official_docs"):
        alignment += 0.15

    return min(1.0, alignment)


def compute_freshness_score(result_item: dict[str, Any], query_has_freshness_intent: bool) -> float:
    """Compute score based on publication date if present, or neutral if missing.
    
    Preserves P3.1 semantics: does not invent dates or penalize missing dates.
    """
    date_str = result_item.get("published_date") or result_item.get("date")
    if not date_str or not isinstance(date_str, str):
        # Neutral score for missing date
        return 0.50

    try:
        # Support ISO timestamps or YYYY-MM-DD
        clean_date = date_str.split("T")[0].strip()
        pub_dt = datetime.strptime(clean_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff_days = (now - pub_dt).days
        if diff_days < 0:
            diff_days = 0

        if diff_days <= 7:
            return 1.00
        elif diff_days <= 30:
            return 0.85
        elif diff_days <= 180:
            return 0.70
        elif diff_days <= 365:
            return 0.55
        else:
            return 0.40
    except Exception:
        return 0.50


FRESHNESS_INTENT_VALUES = {
    "latest",
    "recent",
    "today",
    "yesterday",
    "past_24_hours",
    "past_week",
    "past_month",
    "past_year",
    "news",
    "breaking news",
}

FRESHNESS_INTENT_KEYWORDS = (
    "latest",
    "recent",
    "newest",
    "today",
    "yesterday",
    "breaking news",
    "news",
    "updates",
    "announced",
    "happened",
    "past week",
    "past month",
    "past year",
    "24 hours",
)


def has_freshness_intent(query: str, freshness: str | None = None) -> bool:
    """Determine whether search request has dynamic/current/recency intent."""
    if freshness and freshness.lower().strip() in FRESHNESS_INTENT_VALUES:
        return True
    q_lower = (query or "").lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in FRESHNESS_INTENT_KEYWORDS)


def rank_search_results(
    results: list[dict[str, Any]] | None,
    query: str,
    freshness: str | None = None,
    domain_restriction: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically score, rank, and select search results prior to synthesis.
    
    Parameters:
    - results: List of search result dictionaries (from Tavily, DuckDuckGo, or deduplicator).
    - query: The search query text.
    - freshness: Optional freshness constraint (e.g. 'latest', 'past_week').
    - domain_restriction: Optional domain restriction (e.g. 'github.com').
    - max_results: Max number of top-ranked results to return. If None, returns all ranked results.
    
    Returns:
    List of ranked result dicts with added ranking metadata (`score`, `domain`, `source_type`).
    """
    if not results or not isinstance(results, list):
        return []

    query_is_dynamic = has_freshness_intent(query, freshness)

    # Explainable intent-dependent scoring weights (P4.1):
    # For dynamic / recency-sensitive queries:
    #   relevance: 0.50, authority: 0.25, intent: 0.10, freshness: 0.15
    # For static / reference / timeless technical queries:
    #   relevance: 0.55, authority: 0.30, intent: 0.15, freshness: 0.00
    if query_is_dynamic:
        w_rel = 0.50
        w_auth = 0.25
        w_intent = 0.10
        w_fresh = 0.15
    else:
        w_rel = 0.55
        w_auth = 0.30
        w_intent = 0.15
        w_fresh = 0.00

    scored_items: list[dict[str, Any]] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        raw_url = (item.get("url") or "").strip()
        if not raw_url:
            continue

        canonical = item.get("canonical_url") or canonicalize_url(raw_url)
        if not canonical:
            continue

        title = (item.get("title") or "").strip() or "Untitled result"
        snippet = (item.get("snippet") or item.get("content") or "").strip()

        hostname = extract_hostname(canonical)
        source_type = classify_source_type(hostname, canonical)

        # 1. Compute individual signals
        rel_score = compute_textual_relevance(query, title, snippet)
        auth_score = compute_authority_score(source_type, hostname, canonical)
        intent_score = compute_query_intent_alignment(query, source_type, hostname)
        fresh_score = compute_freshness_score(item, query_is_dynamic)

        # 2. Balanced Composite Score
        composite = (
            (w_rel * rel_score)
            + (w_auth * auth_score)
            + (w_intent * intent_score)
            + (w_fresh * fresh_score)
        )

        entry = dict(item)
        entry["url"] = raw_url  # exact original URL preserved for user citations
        entry["canonical_url"] = canonical
        entry["title"] = title
        entry["snippet"] = snippet
        entry["domain"] = hostname
        entry["source_type"] = source_type
        entry["raw_score"] = round(composite, 4)
        scored_items.append(entry)

    # Sort initially by raw composite score descending
    scored_items.sort(key=lambda x: x["raw_score"], reverse=True)

    # 3. Soft Domain Diversity Penalty
    # Mitigates domain concentration while allowing clearly high-value duplicate domains
    # Do NOT penalize if search was explicitly restricted to a single domain
    is_domain_restricted = bool(domain_restriction)

    domain_counts: dict[str, int] = {}
    ranked_items: list[dict[str, Any]] = []

    for item in scored_items:
        reg_domain = extract_registered_domain(item["domain"])
        count = domain_counts.get(reg_domain, 0)

        penalty = 0.0
        if not is_domain_restricted:
            if count == 1:
                penalty = 0.08
            elif count == 2:
                penalty = 0.18
            elif count >= 3:
                penalty = 0.30

        final_score = max(0.0, item["raw_score"] - penalty)
        item["score"] = round(final_score, 4)
        domain_counts[reg_domain] = count + 1
        ranked_items.append(item)

    # Re-sort with diversity penalty applied
    ranked_items.sort(key=lambda x: x["score"], reverse=True)

    if max_results is not None and max_results > 0:
        return ranked_items[:max_results]
    return ranked_items
