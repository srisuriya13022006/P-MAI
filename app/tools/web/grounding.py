import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from app.tools.web.url import canonicalize_url

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "about", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "it", "its", "that", "this", "these",
    "those", "which", "who", "whom", "what", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "can", "will", "just", "should", "now",
    "from", "into", "also", "here", "there", "their", "them", "then",
}


@dataclass
class EvidenceSource:
    """Represents a structured evidence item extracted from web tools."""
    source_id: str
    title: str
    url: str
    content: str
    final_url: str | None = None
    is_supplementary: bool = False


@dataclass
class GroundingResult:
    """Summary of deterministic factual grounding and citation validation."""
    is_grounded: bool
    contradictions: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    fabricated_urls: list[str] = field(default_factory=list)
    mismatched_citations: list[str] = field(default_factory=list)
    detected_conflicts: list[str] = field(default_factory=list)
    valid_citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_evidence_sources(tool_output: str) -> list[EvidenceSource]:
    """Deterministically parse tool output into structured EvidenceSource items.
    
    Supports:
    - web_search output (<SOURCE_i> blocks and <Tavily_answer> blocks)
    - web_fetch output (<PAGE_SOURCE> blocks)
    """
    if not tool_output or not isinstance(tool_output, str):
        return []

    sources: list[EvidenceSource] = []

    # 1. Parse web_fetch page evidence
    page_match = re.search(
        r"<PAGE_SOURCE>\s*URL:\s*(.*?)\s*\n\s*Final URL:\s*(.*?)\s*\n\s*Title:\s*(.*?)\s*\n\s*Content:\s*([\s\S]*?)</PAGE_SOURCE>",
        tool_output,
        re.IGNORECASE,
    )
    if page_match:
        url = page_match.group(1).strip()
        final_url = page_match.group(2).strip()
        title = page_match.group(3).strip()
        content = page_match.group(4).strip()
        sources.append(
            EvidenceSource(
                source_id="PAGE_SOURCE",
                title=title or "Web Page",
                url=url,
                final_url=final_url if final_url != url else None,
                content=content,
                is_supplementary=False,
            )
        )
        return sources

    # 2. Parse web_search source blocks
    source_pattern = re.compile(
        r"<SOURCE_(\d+)>\s*Title:\s*(.*?)\s*\n\s*URL:\s*(.*?)\s*\n\s*Snippet:\s*([\s\S]*?)</SOURCE_\1>",
        re.IGNORECASE,
    )
    for match in source_pattern.finditer(tool_output):
        idx = match.group(1).strip()
        title = match.group(2).strip()
        url = match.group(3).strip()
        snippet = match.group(4).strip()
        sources.append(
            EvidenceSource(
                source_id=f"SOURCE_{idx}",
                title=title or f"Source {idx}",
                url=url,
                content=snippet,
                is_supplementary=False,
            )
        )

    # 3. Parse supplementary Tavily answer (if present)
    tavily_match = re.search(
        r"<Tavily_answer>([\s\S]*?)</Tavily_answer>",
        tool_output,
        re.IGNORECASE,
    )
    if tavily_match:
        answer_text = tavily_match.group(1).strip()
        if answer_text:
            sources.append(
                EvidenceSource(
                    source_id="TAVILY_ANSWER",
                    title="Tavily Direct Answer",
                    url="",
                    content=answer_text,
                    is_supplementary=True,
                )
            )

    return sources


def extract_markdown_citations(text: str) -> list[tuple[str, str]]:
    """Extract markdown links as (anchor_title, url) tuples."""
    if not text:
        return []
    # Match [Title](https://...)
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")
    return pattern.findall(text)


def detect_fabricated_urls(
    response_text: str,
    sources: list[EvidenceSource],
) -> list[str]:
    """Identify URLs cited in markdown links that were not present in retrieved evidence."""
    citations = extract_markdown_citations(response_text)
    if not citations:
        return []

    valid_urls: set[str] = set()
    for s in sources:
        if s.url:
            valid_urls.add(s.url.strip().lower())
            canon = canonicalize_url(s.url)
            if canon:
                valid_urls.add(canon.lower())
        if s.final_url:
            valid_urls.add(s.final_url.strip().lower())
            canon_final = canonicalize_url(s.final_url)
            if canon_final:
                valid_urls.add(canon_final.lower())

    fabricated: list[str] = []
    for _title, url in citations:
        clean_url = url.strip().lower()
        clean_canon = canonicalize_url(clean_url) or clean_url
        if clean_url not in valid_urls and clean_canon not in valid_urls:
            fabricated.append(url)

    return fabricated


KNOWN_TECH_ENTITIES = {
    "saml", "kerberos", "ldap", "openid", "oauth1", "graphql", "grpc",
    "redis", "rabbitmq", "kafka", "cassandra", "dynamodb", "mongodb",
    "kubernetes", "consul", "vault", "istio"
}

NEGATIVE_EVIDENCE_PATTERNS = [
    r"does not support",
    r"do not support",
    r"not supported",
    r"no support for",
    r"does not use",
    r"do not use",
    r"unavailable",
    r"not compatible with",
    r"lacks support for",
    r"unsupported",
    r"without support for",
    r"cannot use",
]

KNOWN_SOFTWARE_ENTITIES = {
    "python", "fastapi", "cuda", "django", "react", "nodejs", "node",
    "pytorch", "tensorflow", "kubernetes", "postgres", "postgresql", "linux"
}

PREDICTIVE_OR_HISTORICAL_MARKERS = {
    "expected", "upcoming", "planned", "next year", "in the future", "future",
    "preview", "roadmap", "slated", "will be", "legacy", "previously", "retired",
    "deprecated", "historical", "earlier", "prior to", "in development", "alpha", "beta"
}

CONTRADICTION_MARKERS = {
    "latest", "current", "stable", "most recent", "newest", "out now",
    "is released", "is the version", "is available as the latest", "released"
}


def check_entity_support_with_polarity(
    entity: str,
    sources: list[EvidenceSource],
) -> tuple[bool, bool]:
    """Check if a technical entity is supported in evidence and whether negative polarity is present.
    Returns: (is_supported_positively, has_negative_mention).
    A term mentioned only with negative markers (e.g., 'FastAPI does not use SAML') is NOT positive support.
    """
    primary = [s for s in sources if not s.is_supplementary] or sources
    pattern = re.compile(rf"\b{re.escape(entity)}\b", re.IGNORECASE)

    has_positive = False
    has_negative = False

    for s in primary:
        text = s.title + " " + s.content
        for m in pattern.finditer(text):
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            window = text[start:end].lower()
            is_neg = any(re.search(neg, window) for neg in NEGATIVE_EVIDENCE_PATTERNS)
            if is_neg:
                has_negative = True
            else:
                has_positive = True

    return has_positive, has_negative


def clean_unsupported_from_sentence(sentence: str, unsupported: list[str]) -> str:
    """Safely remove unsupported technical entities from sentence lists/clauses (Option A)."""
    cleaned = sentence
    for ent in unsupported:
        pattern1 = re.compile(rf",\s*(?:and|or)\s+{re.escape(ent)}\b", re.IGNORECASE)
        pattern2 = re.compile(rf",\s*{re.escape(ent)}\s*,\s*and\b", re.IGNORECASE)
        pattern3 = re.compile(rf",\s*{re.escape(ent)}\b", re.IGNORECASE)
        pattern4 = re.compile(rf"\b{re.escape(ent)}\s*,\s*", re.IGNORECASE)
        pattern5 = re.compile(rf"\b{re.escape(ent)}\s+(?:and|or)\s+", re.IGNORECASE)
        pattern6 = re.compile(rf"\s+(?:and|or)\s+{re.escape(ent)}\b", re.IGNORECASE)
        # Fallback: standalone entity word (no adjacent comma/conjunction)
        pattern7 = re.compile(rf"\b{re.escape(ent)}\b\s*", re.IGNORECASE)

        if pattern1.search(cleaned):
            cleaned = pattern1.sub("", cleaned)
        elif pattern2.search(cleaned):
            cleaned = pattern2.sub(", and", cleaned)
        elif pattern3.search(cleaned):
            cleaned = pattern3.sub("", cleaned)
        elif pattern4.search(cleaned):
            cleaned = pattern4.sub("", cleaned)
        elif pattern5.search(cleaned):
            cleaned = pattern5.sub("", cleaned)
        elif pattern6.search(cleaned):
            cleaned = pattern6.sub("", cleaned)
        else:
            cleaned = pattern7.sub("", cleaned)

    cleaned = re.sub(r",\s*and\b", " and", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving trailing markdown citations with their preceding statements."""
    raw_splits = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?!\[[^\]]+\]\([^\)]+\))|\n+", text) if s.strip()]
    merged: list[str] = []
    for s in raw_splits:
        if merged and re.fullmatch(r"\[[^\]]+\]\([^\)]+\)", s):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


def is_citation_supported_by_source(
    claim_text: str,
    title: str,
    url: str,
    sources: list[EvidenceSource],
) -> bool:
    """Check if a citation URL exists in evidence and actually supports the claim text.
    Distinguishes citation validity (URL exists) from citation support (source supports claim).
    """
    sources_by_url: dict[str, EvidenceSource] = {}
    for s in sources:
        if s.url:
            sources_by_url[s.url.strip().lower()] = s
            canon = canonicalize_url(s.url)
            if canon:
                sources_by_url[canon.lower()] = s
        if s.final_url:
            sources_by_url[s.final_url.strip().lower()] = s
            canon_f = canonicalize_url(s.final_url)
            if canon_f:
                sources_by_url[canon_f.lower()] = s

    clean_url = url.strip().lower()
    clean_canon = canonicalize_url(clean_url) or clean_url
    source = sources_by_url.get(clean_url) or sources_by_url.get(clean_canon)
    if not source:
        return False

    source_text = (source.title + " " + source.content).lower()

    # Check for version numbers in claim that match source
    version_regex = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")
    claim_versions = set(version_regex.findall(claim_text))
    source_versions = set(version_regex.findall(source_text))
    if claim_versions and (claim_versions & source_versions):
        return True

    # Check for software entities
    claim_lower = claim_text.lower()
    for ent in KNOWN_SOFTWARE_ENTITIES | KNOWN_TECH_ENTITIES:
        if re.search(rf"\b{re.escape(ent)}\b", claim_lower) and re.search(rf"\b{re.escape(ent)}\b", source_text):
            return True

    claim_words = {
        w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", claim_text)
        if w.lower() not in STOPWORDS
    }
    source_words = {
        w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", source_text)
        if w.lower() not in STOPWORDS
    }

    return len(claim_words & source_words) >= 2


def enforce_technical_entities(
    response_text: str,
    sources: list[EvidenceSource],
) -> str:
    """Safely enforce technical entity integrity:
    Option A: Remove unsupported entity when the sentence can be safely reduced.
    Option B: Qualify the claim directly so unsupported entities do not remain asserted facts.
    Citations are only preserved if the cited source actually supports the statement.
    """
    resp_lower = response_text.lower()
    found_entities = [e for e in KNOWN_TECH_ENTITIES if re.search(rf"\b{re.escape(e)}\b", resp_lower)]
    if not found_entities:
        return response_text

    unsupported_map: dict[str, bool] = {}  # entity -> is_negative
    for e in found_entities:
        pos, neg = check_entity_support_with_polarity(e, sources)
        if not pos:
            unsupported_map[e] = neg

    if not unsupported_map:
        return response_text

    sources_by_url: dict[str, EvidenceSource] = {}
    for s in sources:
        if s.url:
            sources_by_url[s.url.strip().lower()] = s
            canon = canonicalize_url(s.url)
            if canon:
                sources_by_url[canon.lower()] = s
        if s.final_url:
            sources_by_url[s.final_url.strip().lower()] = s
            canon_f = canonicalize_url(s.final_url)
            if canon_f:
                sources_by_url[canon_f.lower()] = s

    sentences = split_into_sentences(response_text)
    processed_sentences: list[str] = []

    for sent in sentences:
        sent_unsupported = [e for e in unsupported_map if re.search(rf"\b{re.escape(e)}\b", sent, re.IGNORECASE)]
        if not sent_unsupported:
            processed_sentences.append(sent)
            continue

        # Check if sentence has other supported entities or content words
        has_supported_counterparts = any(
            re.search(rf"\b{w}\b", sent, re.IGNORECASE)
            for w in ["oauth2", "jwt", "bearer", "password", "token", "auth", "session", "cookie", "api", "key", "rest", "http"]
        )

        if has_supported_counterparts:
            reduced = clean_unsupported_from_sentence(sent, sent_unsupported)
            clean_test = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", reduced).strip()
            # Verify the reduced sentence is semantically meaningful, not a hollow stub
            # e.g. "FastAPI uses for authentication" is hollow because the subject is gone
            hollow_stub = bool(re.search(r"\buses\s+for\b", clean_test, re.IGNORECASE))
            clean_words = re.findall(r"\b[a-zA-Z]{3,}\b", clean_test)
            # If reduced sentence has meaningful words (>= 3 words) and is not hollow, use Option A
            if len(clean_words) >= 3 and not hollow_stub:
                processed_sentences.append(reduced)
                continue

        # Option B: Qualify the claim
        # A valid URL must NOT automatically be attached to a newly generated unsupported/qualified statement
        # Only retain citation if the cited source actually mentions the unsupported entity
        citations = extract_markdown_citations(sent)
        supported_citations = []
        for c_title, c_url in citations:
            clean_u = canonicalize_url(c_url.strip().lower()) or c_url.strip().lower()
            src = sources_by_url.get(clean_u) or sources_by_url.get(c_url.strip().lower())
            if src:
                src_text = (src.title + " " + src.content).lower()
                if any(re.search(rf"\b{re.escape(e)}\b", src_text) for e in sent_unsupported):
                    supported_citations.append(f"[{c_title}]({c_url})")

        cit_str = " " + " ".join(supported_citations) if supported_citations else ""
        neg_items = [e for e in sent_unsupported if unsupported_map[e]]
        absent_items = [e for e in sent_unsupported if not unsupported_map[e]]

        notes = []
        if neg_items:
            items_str = ", ".join(sorted(neg_items))
            notes.append(f"The retrieved sources indicate that {items_str} is not supported")
        if absent_items:
            items_str = ", ".join(sorted(absent_items))
            notes.append(f"The retrieved sources do not establish support for {items_str}")

        qualified = ("; ".join(notes) + f".{cit_str}").rstrip()
        processed_sentences.append(qualified)

    return " ".join(processed_sentences)


def extract_entity_versions_from_sources(sources: list[EvidenceSource]) -> dict[str, set[str]]:
    """Extract known software entities and their associated versions from evidence."""
    primary = [s for s in sources if not s.is_supplementary] or sources
    full_text = " ".join(s.title + " " + s.content for s in primary)

    ev_map: dict[str, set[str]] = {}
    for ent in KNOWN_SOFTWARE_ENTITIES:
        v_matches = re.findall(rf"\b{re.escape(ent)}\b(?:[^\.\n\?!]{{0,30}}?)\b(\d+\.\d+(?:\.\d+)?)\b", full_text, re.IGNORECASE)
        v_matches += re.findall(rf"\b(\d+\.\d+(?:\.\d+)?)\b(?:[^\.\n\?!]{{0,30}}?)\b{re.escape(ent)}\b", full_text, re.IGNORECASE)
        if v_matches:
            ev_map[ent.lower()] = set(v_matches)

    return ev_map


def enforce_version_contradictions_safe(
    response_text: str,
    sources: list[EvidenceSource],
) -> str:
    """Safely enforce version contradiction correctness without blanket regex replacements:
    1. Associates versions with their specific entities (Python, FastAPI, CUDA).
    2. Does not touch predictive, future, or historical statements.
    3. Does not corrupt unrelated version statements across clauses or sentences.
    4. Rewrites directly contradictory claims to reflect evidence-established versions.
    """
    primary = [s for s in sources if not s.is_supplementary] or sources
    ev_map = extract_entity_versions_from_sources(primary)

    version_regex = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")
    all_evidence_versions = set(version_regex.findall(" ".join(s.content + " " + s.title for s in primary)))

    sentences = split_into_sentences(response_text)
    processed_sentences: list[str] = []
    clause_delimiters = r"[,;]|\b(?:while|whereas|but|and|although)\b"

    for sent in sentences:
        clauses = [c.strip() for c in re.split(clause_delimiters, sent) if c.strip()]
        new_sent = sent
        sentence_modified = False

        # 1. Entity-associated version contradiction check
        for ent, ev_versions in ev_map.items():
            if not ev_versions or sentence_modified:
                continue
            best_established = max(ev_versions, key=len)
            ent_title = ent.capitalize() if ent != "fastapi" else "FastAPI"
            if ent == "cuda":
                ent_title = "CUDA"

            for clause in clauses:
                if not re.search(rf"\b{re.escape(ent)}\b", clause, re.IGNORECASE):
                    continue

                v_matches = re.findall(rf"\b{re.escape(ent)}\b(?:[^\.\n\?!]{{0,30}}?)\b(\d+\.\d+(?:\.\d+)?)\b", clause, re.IGNORECASE)
                v_matches += re.findall(rf"\b(\d+\.\d+(?:\.\d+)?)\b(?:[^\.\n\?!]{{0,30}}?)\b{re.escape(ent)}\b", clause, re.IGNORECASE)

                clause_lower = clause.lower()
                is_predictive = any(m in clause_lower for m in PREDICTIVE_OR_HISTORICAL_MARKERS)
                is_contradictory = any(m in clause_lower for m in CONTRADICTION_MARKERS)

                for v in set(v_matches):
                    if v in ev_versions:
                        continue  # Supported version
                    if is_predictive and not is_contradictory:
                        continue  # Predictive or historical context

                    if is_contradictory:
                        # Direct contradiction for this entity in this clause!
                        replacement_clause = f"The retrieved sources identify {ent_title} {best_established} as the latest stable release"
                        clean_sent_no_links = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", sent).strip()
                        clause_no_links = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", clause).strip()
                        if clause_no_links in clean_sent_no_links and len(clean_sent_no_links) <= len(clause_no_links) + 10:
                            citations = extract_markdown_citations(sent)
                            supported_citations = [
                                f"[{c_title}]({c_url})" for c_title, c_url in citations
                                if is_citation_supported_by_source(f"{ent_title} {best_established}", c_title, c_url, sources)
                            ]
                            cit_str = " " + " ".join(supported_citations) if supported_citations else ""
                            new_sent = f"{replacement_clause}.{cit_str}".rstrip()
                        else:
                            new_sent = new_sent.replace(clause, replacement_clause)
                        sentence_modified = True
                        break
                if sentence_modified:
                    break

        # 2. Fallback for entity-agnostic direct contradiction (e.g. '3.15 is the latest stable version')
        if not sentence_modified and all_evidence_versions:
            sent_versions = set(version_regex.findall(sent))
            for v in sent_versions:
                if v not in all_evidence_versions:
                    v_major = v.split(".")[0]
                    matching = [ev for ev in all_evidence_versions if ev.startswith(v_major + ".")]
                    if matching:
                        sent_lower = sent.lower()
                        is_predictive = any(m in sent_lower for m in PREDICTIVE_OR_HISTORICAL_MARKERS)
                        is_contradictory = any(m in sent_lower for m in CONTRADICTION_MARKERS)
                        if is_contradictory and not is_predictive:
                            best_v = max(matching, key=len)
                            citations = extract_markdown_citations(sent)
                            supported_citations = [
                                f"[{c_title}]({c_url})" for c_title, c_url in citations
                                if is_citation_supported_by_source(f"version {best_v}", c_title, c_url, sources)
                            ]
                            cit_str = " " + " ".join(supported_citations) if supported_citations else ""
                            new_sent = f"The retrieved sources identify {best_v} as the latest stable release.{cit_str}".rstrip()
                            sentence_modified = True
                            break

        processed_sentences.append(new_sent)

    return " ".join(processed_sentences)


def detect_version_contradictions(
    response_text: str,
    sources: list[EvidenceSource],
) -> list[dict[str, Any]]:
    """Detect version claims in response that contradict retrieved primary evidence.
    Returns list of contradiction detail dicts with keys:
    - unsupported_version: version in response
    - established_versions: versions present in primary evidence
    - entity_prefix: major version family
    Predictive/historical version mentions are not flagged as contradictions.
    """
    version_regex = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")

    primary_sources = [s for s in sources if not s.is_supplementary] or sources
    evidence_text = " ".join(s.content + " " + s.title for s in primary_sources)

    evidence_versions = set(version_regex.findall(evidence_text))
    response_versions = set(version_regex.findall(response_text))

    contradictions: list[dict[str, Any]] = []
    sentences = split_into_sentences(response_text)

    for v_resp in response_versions:
        if v_resp not in evidence_versions:
            v_sentences = [s for s in sentences if v_resp in s]
            if v_sentences:
                all_predictive = True
                for s in v_sentences:
                    s_lower = s.lower()
                    if not any(m in s_lower for m in PREDICTIVE_OR_HISTORICAL_MARKERS):
                        all_predictive = False
                        break
                if all_predictive:
                    continue

            resp_major = v_resp.split(".")[0]
            matching_major_evid = [ev for ev in evidence_versions if ev.startswith(resp_major + ".")]
            if matching_major_evid:
                contradictions.append({
                    "unsupported_version": v_resp,
                    "established_versions": sorted(matching_major_evid),
                    "best_established": max(matching_major_evid, key=len),
                    "message": (
                        f"Response asserts version '{v_resp}' which is absent from evidence "
                        f"(evidence establishes: {', '.join(sorted(matching_major_evid))})."
                    ),
                })

    return contradictions


def detect_evidence_conflicts(sources: list[EvidenceSource]) -> list[str]:
    """Detect factual disagreements across retrieved sources (e.g., date conflicts)."""
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    conflicts: list[str] = []
    primary = [s for s in sources if not s.is_supplementary]
    if len(primary) < 2:
        return []

    # Check for month-level launch/release date discrepancies for similar subjects
    found_months_by_source: dict[str, set[str]] = {}
    for s in primary:
        text_lower = (s.title + " " + s.content).lower()
        present_months = {m for m in months if re.search(r"\b" + m + r"\b", text_lower)}
        if present_months:
            found_months_by_source[s.source_id] = present_months

    if len(found_months_by_source) >= 2:
        all_month_sets = list(found_months_by_source.values())
        first_set = all_month_sets[0]
        if any(ms != first_set and not (ms & first_set) for ms in all_month_sets[1:]):
            distinct_groups = []
            seen_groups = []
            for ms in all_month_sets:
                sorted_ms = tuple(sorted(ms))
                if sorted_ms not in seen_groups:
                    seen_groups.append(sorted_ms)
                    distinct_groups.append(", ".join(m.capitalize() for m in sorted_ms))

            if len(distinct_groups) == 2:
                conflict_msg = f"The retrieved sources report conflicting dates or months: some report {distinct_groups[0]}, while others report {distinct_groups[1]}."
            else:
                conflict_msg = f"The retrieved sources report conflicting dates or months: some report {distinct_groups[0]}, while others report {' or '.join(distinct_groups[1:])}."
            conflicts.append(conflict_msg)

    return conflicts


def check_unsupported_technical_entities(
    response_text: str,
    sources: list[EvidenceSource],
) -> list[str]:
    """Detect distinct technical entities introduced in the response that have zero support in evidence
    or contradict negative evidence in sources.
    """
    response_lower = response_text.lower()
    unsupported: list[str] = []

    for entity in KNOWN_TECH_ENTITIES:
        if re.search(r"\b" + entity + r"\b", response_lower):
            pos, neg = check_entity_support_with_polarity(entity, sources)
            if not pos:
                if neg:
                    unsupported.append(f"Technical entity '{entity}' appears in response despite negative evidence indicating it is not supported.")
                else:
                    unsupported.append(f"Technical entity '{entity}' appears in response without supporting evidence.")

    return unsupported


def check_citation_source_support(
    response_text: str,
    sources: list[EvidenceSource],
) -> list[tuple[str, str, str]]:
    """Detect citations where the cited source has zero keyword overlap with the sentence citing it.
    Returns list of (claim_sentence, anchor_title, cited_url).
    """
    citations = extract_markdown_citations(response_text)
    if not citations:
        return []

    sources_by_url: dict[str, EvidenceSource] = {}
    for s in sources:
        if s.url:
            sources_by_url[s.url.strip().lower()] = s
            canon = canonicalize_url(s.url)
            if canon:
                sources_by_url[canon.lower()] = s

    # Split response into sentences
    sentences = split_into_sentences(response_text)
    mismatched: list[tuple[str, str, str]] = []

    def _stem(word: str) -> str:
        w = word.lower()
        if len(w) > 4:
            if w.endswith("ing"):
                return w[:-3]
            if w.endswith("ed") or w.endswith("es"):
                return w[:-2]
            if w.endswith("s"):
                return w[:-1]
        return w

    for title, url in citations:
        clean_url = url.strip().lower()
        clean_canon = canonicalize_url(clean_url) or clean_url
        source = sources_by_url.get(clean_url) or sources_by_url.get(clean_canon)
        if not source:
            continue

        # Find sentence containing this citation
        sent_idx = -1
        for i, s in enumerate(sentences):
            if url in s:
                sent_idx = i
                break
        if sent_idx == -1:
            continue

        sent = sentences[sent_idx]

        # Text in current sentence before the citation
        before_citation = sent.split(url)[0]
        before_clean = re.sub(r"\[[^\]]*\]\(?", "", before_citation).strip()
        words_before = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", before_clean.lower()) if w not in STOPWORDS]

        # Context words to check:
        # If words_before >= 2, the claim is right before the citation in this sentence.
        # If words_before < 2 (e.g. citation follows a period or starts the sentence), check preceding sentence as well.
        candidate_texts = []
        if len(words_before) >= 2:
            candidate_texts.append(before_clean)
        else:
            if sent_idx > 0:
                candidate_texts.append(sentences[sent_idx - 1])
            after_citation = sent.split(url, 1)[1] if url in sent else ""
            after_clean = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", after_citation).strip()
            if after_clean:
                candidate_texts.append(after_clean)

        combined_claim = " ".join(candidate_texts)
        combined_claim = re.sub(r"\(unverified citation removed\)", "", combined_claim, flags=re.IGNORECASE)
        claim_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", combined_claim.lower()) if w not in STOPWORDS]
        claim_stems = {_stem(w) for w in claim_words if len(w) >= 3}

        source_raw = (source.title + " " + source.content).lower()
        source_stems = {_stem(w) for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", source_raw) if w not in STOPWORDS}

        # If claim has significant content tokens but zero overlap with cited source, flag mismatch
        if len(claim_stems) >= 2 and len(claim_stems & source_stems) == 0:
            mismatched.append((sent, title, url))

    return mismatched


def verify_grounding(
    response_text: str,
    tool_output: str,
    user_query: str = "",
) -> GroundingResult:
    """Perform deterministic claim-to-source grounding and citation verification."""
    sources = parse_evidence_sources(tool_output)

    fabricated = detect_fabricated_urls(response_text, sources)
    version_contradictions = detect_version_contradictions(response_text, sources)
    contradiction_msgs = [c["message"] for c in version_contradictions]
    unsupported_entities = check_unsupported_technical_entities(response_text, sources)
    conflicts = detect_evidence_conflicts(sources)
    mismatches = check_citation_source_support(response_text, sources)
    mismatch_urls = [url for _sent, _t, url in mismatches]

    citations = extract_markdown_citations(response_text)
    valid_citations = [url for _title, url in citations if url not in fabricated and url not in mismatch_urls]

    is_grounded = not fabricated and not contradiction_msgs and not unsupported_entities and not mismatches

    return GroundingResult(
        is_grounded=is_grounded,
        contradictions=contradiction_msgs,
        unsupported_claims=unsupported_entities,
        fabricated_urls=fabricated,
        mismatched_citations=mismatch_urls,
        detected_conflicts=conflicts,
        valid_citations=valid_citations,
    )


def enforce_grounding(
    response_text: str,
    tool_output: str,
    user_query: str = "",
) -> str:
    """Enforce deterministic grounding and citation integrity:
    1. Empty/unusable evidence -> returns honest insufficiency statement.
    2. Version contradictions -> replaces contradictory version with evidence-established version.
    3. Unsupported technical entities -> qualifies that sources do not support them.
    4. Fabricated citations -> converts to non-citation label / removes link markup so fake URLs do not survive.
    5. Mismatched citations -> removes unrelated source links from unsupported sentences.
    6. Conflicting evidence -> ensures conflict between sources is acknowledged.
    7. Strips any leaked internal evidence tags.
    """
    if not response_text:
        return ""

    sources = parse_evidence_sources(tool_output)
    primary_sources = [s for s in sources if not s.is_supplementary]

    # 1. Empty or unusable evidence check
    if not sources or (not primary_sources and not any(s.content.strip() for s in sources)):
        return "The retrieved sources did not provide enough information to answer reliably."

    text = response_text

    # 2. Strip leaked internal evidence tags
    internal_tags = [
        r"</?SOURCE_\d+>",
        r"</?search_evidence>",
        r"</?page_evidence>",
        r"</?PAGE_SOURCE>",
        r"</?Tavily_answer>",
    ]
    for tag_pattern in internal_tags:
        text = re.sub(tag_pattern, "", text, flags=re.IGNORECASE)

    # 3. Fabricated Citation Enforcement:
    # Only validated evidence URLs may appear as links.
    # Convert fabricated links to clear non-citation label or remove link markup entirely.
    valid_urls: set[str] = set()
    for s in sources:
        if s.url:
            valid_urls.add(s.url.strip().lower())
            canon = canonicalize_url(s.url)
            if canon:
                valid_urls.add(canon.lower())
        if s.final_url:
            valid_urls.add(s.final_url.strip().lower())
            canon_final = canonicalize_url(s.final_url)
            if canon_final:
                valid_urls.add(canon_final.lower())

    def _replace_fabricated_link(match):
        title = match.group(1).strip()
        url = match.group(2).strip()
        clean_url = url.lower()
        clean_canon = canonicalize_url(clean_url) or clean_url
        if clean_url in valid_urls or clean_canon in valid_urls:
            return match.group(0)  # Keep valid citation
        # Convert fabricated citation into clearly non-citation text
        return f"{title} (unverified citation removed)"

    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", _replace_fabricated_link, text)

    # 4. Mismatched Citation Enforcement:
    # Remove citations attached to claims that have zero support in the cited source.
    mismatches = check_citation_source_support(text, sources)
    for _claim, title, url in mismatches:
        # De-link the mismatched citation from the unsupported sentence
        text = text.replace(f"[{title}]({url})", title)

    # 5. Unsupported Version Contradiction Enforcement (Safe & Context-Aware):
    text = enforce_version_contradictions_safe(text, sources)

    # 6. Unsupported Technical Entity Enforcement (Option A Reduction or Option B Qualification):
    text = enforce_technical_entities(text, sources)

    # 7. Conflicting Evidence Qualification:
    conflicts = detect_evidence_conflicts(sources)
    if conflicts:
        conflict_terms = ["disagree", "conflict", "differ", "on the other hand", "whereas", "while"]
        if not any(term in text.lower() for term in conflict_terms):
            text += f"\n\nNote on conflicting sources: {conflicts[0]}"

    # 8. Final Sanitization of Internal Implementation Markers:
    internal_implementation_patterns = [
        r"</?(?:SOURCE_\d+|PAGE_SOURCE|search_evidence|page_evidence|Tavily_answer)[^>]*>",
        r"\[?\s*\bSOURCE_\d+\b\s*\]?:?",
        r"\(\s*\bSOURCE_\d+\b\s*\):?",
        r"\bSOURCE_\d+\b:?",
        r"\bPAGE_SOURCE\b",
        r"\bTavily_answer\b",
        r"【\s*SOURCE_\d+\s*】",
    ]
    for p in internal_implementation_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # Clean up empty brackets or excess whitespace
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Final safety assertion: Internal SOURCE_* identifiers must NEVER leak
    assert not re.search(r"\bSOURCE_\d+\b", text, flags=re.IGNORECASE), "Internal SOURCE_* identifier leaked to final output!"

    return text


def sanitize_grounded_response(
    response_text: str,
    tool_output: str,
) -> str:
    """Enforce grounding and sanitize response (backward-compatible alias)."""
    return enforce_grounding(response_text, tool_output)
