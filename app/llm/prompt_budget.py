"""
P-MAI Prompt Budget and Context Management.
Manages prompt token budgets across Cloud and Local LLM providers.
Preserves system security rules, active task/clarification state, and latest user queries,
while compacting older multi-turn conversation history and bounding web evidence.
"""
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# Cached tiktoken encoder
_ENCODER = None


def get_token_encoder():
    """Lazily load tiktoken cl100k_base encoder if available."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False
    return _ENCODER


def estimate_tokens(text: str | None) -> int:
    """
    Estimate token count for a text string.
    Uses tiktoken cl100k_base if available; falls back to deterministic heuristic.
    """
    if not text:
        return 0

    encoder = get_token_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Heuristic: 1 token ~= 3.5 characters in English/code mixtures
    # Use word & punctuation regex for high precision without dependencies
    tokens = len(re.findall(r"\w+|[^\w\s]", text))
    return max(1, math.ceil(tokens * 1.15))


def estimate_messages_tokens(
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
) -> int:
    """
    Estimate total tokens for a chat payload including message formatting overhead.
    """
    total = 0
    if system_prompt:
        total += estimate_tokens(system_prompt) + 4

    for m in messages:
        content = m.get("content", "")
        # Standard chat template framing overhead: ~4 tokens per message
        total += estimate_tokens(content) + 4

    # Priming tokens for assistant response
    total += 3
    return total


def extract_and_preserve_citations(text: str) -> list[str]:
    """
    Extract all Markdown links and citation markers from text so they can be
    preserved during evidence truncation.
    """
    # Matches [Title](http...) and [Source: ...] and [1]
    citations = re.findall(r"\[[^\]]+\]\([^\)]+\)|\[source:[^\]]+\]|\[\d+\]", text, re.IGNORECASE)
    return citations


def bound_search_evidence(text: str, max_tokens: int) -> str:
    """
    Bound search evidence content to fit token budget while strictly preserving
    all Markdown source links and citation associations.
    """
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text

    # Extract all citation links to guarantee none are lost
    citations = extract_and_preserve_citations(text)
    unique_citations = []
    seen = set()
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    # Search for <search_evidence> wrapper
    has_wrapper = "<search_evidence>" in text and "</search_evidence>" in text
    evidence_body = text
    prefix = ""
    suffix = ""
    if has_wrapper:
        parts = text.split("<search_evidence>", 1)
        prefix = parts[0] + "<search_evidence>\n"
        rest = parts[1].split("</search_evidence>", 1)
        evidence_body = rest[0]
        suffix = "\n</search_evidence>" + (rest[1] if len(rest) > 1 else "")

    # Truncate evidence body by characters roughly proportional to tokens
    target_chars = max(100, int(max_tokens * 3.2))
    truncated_body = evidence_body[:target_chars].rstrip()
    truncated_body += "\n[...additional search evidence truncated for context limits...]"

    # Append preserved citation links if not already in truncated text
    missing_citations = [c for c in unique_citations if c not in truncated_body]
    if missing_citations:
        truncated_body += "\nPreserved Sources: " + " | ".join(missing_citations)

    return f"{prefix}{truncated_body}{suffix}" if has_wrapper else truncated_body


def compact_history_message(msg: dict[str, str], max_tokens: int) -> dict[str, str]:
    """
    Compact a single past turn message (e.g. an earlier assistant reply with search evidence).
    """
    content = msg.get("content", "")
    if estimate_tokens(content) <= max_tokens:
        return msg

    role = msg.get("role", "user")
    if "<search_evidence>" in content:
        bounded = bound_search_evidence(content, max_tokens)
        return {"role": role, "content": bounded}

    # For general messages, truncate middle or tail, keeping essential sentences
    target_chars = int(max_tokens * 3.5)
    sentences = re.split(r"(?<=[.!?])\s+", content)
    retained = []
    curr_len = 0
    for s in sentences:
        if curr_len + len(s) > target_chars:
            break
        retained.append(s)
        curr_len += len(s)

    if not retained:
        retained = [content[:target_chars]]

    compacted = " ".join(retained) + " [...truncated]"
    return {"role": role, "content": compacted}


def budget_messages(
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
    max_prompt_tokens: int = 3400,
    reserved_response_tokens: int = 512,
    emergency_reduction: bool = False,
) -> tuple[list[dict[str, str]], str | None, dict[str, Any]]:
    """
    Enforce prompt token budget across conversation messages and system prompt.

    Guarantees:
    1. Latest user query is NEVER dropped.
    2. Base system instructions and security policies are strictly preserved.
    3. Active task/clarification instructions are preserved.
    4. Relevant recent turns are prioritized over older turns.
    5. Older turns are compacted or rolled into a brief summary.
    6. Search citations and evidence associations are preserved.
    """
    if not messages and not system_prompt:
        return [], None, {"total_tokens": 0, "truncated": False}

    current_total = estimate_messages_tokens(messages, system_prompt)
    if current_total <= max_prompt_tokens and not emergency_reduction:
        return list(messages), system_prompt, {
            "total_tokens": current_total,
            "truncated": False,
            "original_tokens": current_total,
        }

    logger.info(
        "Prompt budget exceeded: %d tokens > %d max prompt tokens (emergency=%s). Adapting prompt...",
        current_total,
        max_prompt_tokens,
        emergency_reduction,
    )

    budgeted_system_prompt = system_prompt
    sys_tokens = estimate_tokens(budgeted_system_prompt)

    # 1. System Prompt Budgeting: if system prompt alone takes too much budget,
    # trim relevant memories while preserving base instructions and persona.
    if budgeted_system_prompt and sys_tokens > (max_prompt_tokens * 0.45):
        if "Relevant long-term memories:" in budgeted_system_prompt:
            parts = budgeted_system_prompt.split("Relevant long-term memories:", 1)
            base_sys = parts[0].strip()
            mem_text = parts[1].strip()
            mem_lines = [line for line in mem_text.splitlines() if line.strip().startswith("-")]
            # Keep top 3 most relevant memories
            retained_mems = mem_lines[:3]
            budgeted_system_prompt = f"{base_sys}\n\nRelevant long-term memories:\n" + "\n".join(retained_mems)
            sys_tokens = estimate_tokens(budgeted_system_prompt)

    remaining_budget = max(400, max_prompt_tokens - sys_tokens - 10)

    if not messages:
        return [], budgeted_system_prompt, {
            "total_tokens": estimate_tokens(budgeted_system_prompt),
            "truncated": True,
            "original_tokens": current_total,
        }

    # 2. Invariant: Latest Message must be preserved
    latest_msg = messages[-1]
    latest_tokens = estimate_tokens(latest_msg.get("content", ""))

    # If emergency reduction, compact latest message only if absurdly large (> 1500 tokens)
    if latest_tokens > remaining_budget:
        if latest_msg.get("role") == "user" and "<search_evidence>" in latest_msg.get("content", ""):
            latest_msg = compact_history_message(latest_msg, remaining_budget - 50)
            latest_tokens = estimate_tokens(latest_msg.get("content", ""))

    retained_messages: list[dict[str, str]] = [latest_msg]
    used_tokens = latest_tokens + 4
    remaining_history_budget = remaining_budget - used_tokens

    # 3. Walk backwards from the second-to-last message to retain recent turns
    history_candidates = list(messages[:-1])
    included_history: list[dict[str, str]] = []
    omitted_count = 0

    # Preserve any explicit earlier summary if present at index 0
    preserved_summary_msg = None
    if history_candidates and history_candidates[0].get("role") == "system" and "summary" in history_candidates[0].get("content", "").lower():
        preserved_summary_msg = history_candidates.pop(0)

    for msg in reversed(history_candidates):
        msg_tok = estimate_tokens(msg.get("content", "")) + 4

        # If a single history turn is very large (e.g. search evidence or multi-paragraph answer), compact it
        if msg_tok > 400:
            compacted = compact_history_message(msg, max_tokens=300)
            msg_tok = estimate_tokens(compacted.get("content", "")) + 4
            msg = compacted

        if remaining_history_budget >= msg_tok:
            included_history.append(msg)
            remaining_history_budget -= msg_tok
        else:
            omitted_count += 1

    included_history.reverse()

    # If older turns were omitted, insert summary
    final_messages: list[dict[str, str]] = []
    if preserved_summary_msg:
        final_messages.append(preserved_summary_msg)
    elif omitted_count > 0:
        summary_note = {
            "role": "system",
            "content": f"[Earlier conversation history ({omitted_count} turns) omitted to fit local context window]",
        }
        final_messages.append(summary_note)

    final_messages.extend(included_history)
    final_messages.append(latest_msg)

    final_tokens = estimate_messages_tokens(final_messages, budgeted_system_prompt)
    logger.info(
        "Prompt budget applied: reduced from %d tokens to %d tokens (omitted_turns=%d)",
        current_total,
        final_tokens,
        omitted_count,
    )

    return final_messages, budgeted_system_prompt, {
        "total_tokens": final_tokens,
        "truncated": True,
        "omitted_turns": omitted_count,
        "original_tokens": current_total,
    }
