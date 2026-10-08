"""URL canonicalization and web search result deduplication utilities."""

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Known marketing, tracking, and campaign parameters to strip for canonical identity
TRACKING_QUERY_PARAMS = {
    # Google Analytics / Urchin
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    # Ad and Click IDs
    "gclid",
    "fbclid",
    "msclkid",
    "dclid",
    "twclid",
    "yclid",
    # Platform trackers
    "_ga",
    "_gl",
    "mc_cid",
    "mc_eid",
    "igshid",
}


def canonicalize_url(url: str) -> str:
    """Deterministically canonicalize a search-result URL for identity and deduplication.

    - Validates scheme is HTTP or HTTPS (rejects javascript, ftp, file, etc.).
    - Normalizes scheme and host to lowercase.
    - Strips standard default ports (:80 for http, :443 for https).
    - Removes fragments (#section).
    - Filters known tracking query parameters (utm_*, gclid, fbclid, etc.).
    - Preserves meaningful query parameters (?page=2, ?version=3.14, etc.) in sorted order.
    - Normalizes trailing slashes on paths.

    Returns the canonical URL string, or empty string "" if the URL is invalid or unsafe.
    """
    if not url or not isinstance(url, str):
        return ""

    raw = url.strip()
    if not raw:
        return ""

    lower_raw = raw.lower()
    if lower_raw.startswith(("javascript:", "data:", "ftp:", "file:", "mailto:", "tel:")):
        return ""

    try:
        parsed = urlparse(raw)
    except Exception:
        return ""

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    netloc = parsed.netloc.lower()
    if not netloc:
        return ""

    # Strip default ports if present
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize path (remove trailing slash except for root empty path)
    path = parsed.path
    if path:
        path = path.rstrip("/")

    # Normalize query: strip tracking params, sort remaining meaningful params
    clean_query = ""
    if parsed.query:
        try:
            query_items = parse_qsl(parsed.query, keep_blank_values=True)
            filtered = [
                (k, v) for k, v in query_items
                if k.lower() not in TRACKING_QUERY_PARAMS
            ]
            if filtered:
                filtered.sort(key=lambda item: (item[0], item[1]))
                clean_query = urlencode(filtered)
        except Exception:
            clean_query = ""

    # Fragments are stripped for canonical identity
    fragment = ""

    return urlunparse((scheme, netloc, path, "", clean_query, fragment))


def is_valid_web_url(url: str) -> bool:
    """Return True if the URL is a valid HTTP/HTTPS web URL."""
    return bool(canonicalize_url(url))


def is_valid_domain(domain: str) -> bool:
    """Validate that a string is a well-formed public domain (e.g. 'github.com', 'docs.python.org').

    Rejects IP addresses, schemes, paths, ports, and internal/loopback names.
    """
    if not domain or not isinstance(domain, str):
        return False
    clean = domain.strip().lower()
    # Strip potential http:// or https://
    if "://" in clean:
        clean = clean.split("://", 1)[1]
    clean = clean.split("/", 1)[0].split(":", 1)[0].strip()
    if not clean:
        return False

    # Block loopback, local hostnames, or missing dots
    if clean in ("localhost", "ip6-localhost", "ip6-loopback") or "." not in clean:
        return False

    # Block IP literals
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", clean) or ":" in clean:
        return False

    # Standard RFC domain label regex:
    # 2 or more labels, each 1-63 chars of letters/digits/hyphens, ending in alpha TLD >= 2 chars
    domain_regex = r"^(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
    return bool(re.match(domain_regex, clean))



def deduplicate_search_results(
    raw_results: list[dict[str, Any]] | None,
    max_results: int | None = 5,
    max_snippet_len: int = 600,
) -> list[dict[str, Any]]:
    """Deduplicate raw web search results by canonical URL while preserving original URLs and provider order.

    - Rejects invalid or unsafe URLs.
    - Collapses duplicate canonical URLs into a single source.
    - Conservatively prefers better title/snippet metadata when duplicates exist.
    - Preserves exact original_url for citations.
    - Returns up to max_results unique sources (or all if max_results is None).
    """
    if not raw_results or not isinstance(raw_results, list):
        return []

    valid_sources: list[dict[str, Any]] = []
    seen_canonical: dict[str, int] = {}

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        orig_url = (item.get("url") or "").strip()
        canonical = canonicalize_url(orig_url)
        if not canonical:
            continue

        raw_title = (item.get("title") or "").strip()
        title = raw_title if raw_title else "Untitled result"
        snippet = (item.get("snippet") or item.get("content") or "").strip()
        if len(snippet) > max_snippet_len:
            snippet = snippet[:max_snippet_len].rstrip() + "..."

        if canonical in seen_canonical:
            # Result is a duplicate of an earlier result; conservatively upgrade metadata if previous was low quality
            existing_idx = seen_canonical[canonical]
            existing = valid_sources[existing_idx]
            if existing["title"] in ("Untitled result", "Result", "") and title not in ("Untitled result", "Result", ""):
                existing["title"] = title
            if len(existing["snippet"]) < 20 and len(snippet) >= 20:
                existing["snippet"] = snippet
            continue

        if max_results is not None and len(valid_sources) >= max_results:
            break

        source_entry = dict(item)
        source_entry["title"] = title
        source_entry["url"] = orig_url  # exact original URL preserved for user citations
        source_entry["canonical_url"] = canonical
        source_entry["snippet"] = snippet
        seen_canonical[canonical] = len(valid_sources)
        valid_sources.append(source_entry)

    return valid_sources
