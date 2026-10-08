"""Conflict detection module for MAI long-term memories.

Distinguishes between:
  1. Duplicate  -> exact or near-identical wording (handled prior to conflict check).
  2. Conflict   -> same memory type and subject/domain, but different active values.
  3. New Memory -> different type or independent subject/domain.

Conflict Rules:
  - Facts: Collision on the same attribute slot (e.g. user's name, user's city/location).
  - Preferences: Mutually exclusive alternatives within the same domain
    (e.g. programming languages, theme modes, editors, indent styles).
  - Goals: Mutually exclusive goals or frameworks in the same domain.
  - Projects: Updates to the same named project entity (e.g. "P-MAI").
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database.models import Memory
    from app.schemas.memory import MemoryCandidate

# Mutually exclusive domain clusters for preferences and goals
_DOMAIN_CLUSTERS: list[set[str]] = [
    # Programming languages
    {"python", "javascript", "typescript", "c++", "cpp", "rust", "go", "golang", "java", "c#", "csharp", "php", "ruby", "kotlin", "swift", "scala", "c"},
    # UI Themes & Modes
    {"dark mode", "light mode", "dark theme", "light theme", "dark", "light"},
    # Indentation
    {"tabs", "spaces", "tab", "space"},
    # ML / DL Frameworks
    {"tensorflow", "pytorch", "keras", "jax", "scikit-learn", "sklearn"},
    # Web Frameworks
    {"fastapi", "django", "flask", "express", "next.js", "nextjs", "react", "vue", "angular", "svelte"},
    # Editors & IDEs
    {"vim", "neovim", "emacs", "vscode", "vs code", "pycharm", "intellij", "sublime"},
    # Operating Systems
    {"linux", "windows", "macos", "mac os", "ubuntu"},
]

# Patterns for identifying fact slots
_FACT_SLOTS: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"\b(?:name is|called)\b", re.IGNORECASE)),
    ("location", re.compile(r"\b(?:lives in|living in|located in|based in|from)\b", re.IGNORECASE)),
    ("age", re.compile(r"\b(?:years old|age is)\b", re.IGNORECASE)),
    ("role", re.compile(r"\b(?:works as|employed as|job is|occupation is)\b", re.IGNORECASE)),
]


class ConflictDetector:
    """Detects whether a memory candidate conflicts with an existing active memory."""

    @classmethod
    def find_conflict(
        cls,
        candidate: MemoryCandidate,
        existing_memories: list[Memory],
    ) -> Memory | None:
        """Return the first existing active memory that conflicts with candidate, or None."""
        for memory in existing_memories:
            if not getattr(memory, "is_active", True):
                continue
            if memory.memory_type != candidate.memory_type:
                continue
            if cls._is_conflict(candidate, memory):
                return memory
        return None

    @classmethod
    def _is_conflict(cls, candidate: MemoryCandidate, existing: Memory) -> bool:
        c_text = candidate.content.strip().lower()
        e_text = existing.content.strip().lower()

        # If identical, it's a duplicate, not a conflict (duplicate handler handles this)
        if c_text == e_text:
            return False

        m_type = candidate.memory_type

        if m_type == "fact":
            return cls._is_fact_conflict(c_text, e_text)
        elif m_type == "preference":
            return cls._is_preference_conflict(c_text, e_text)
        elif m_type == "goal":
            return cls._is_goal_conflict(c_text, e_text)
        elif m_type == "project":
            return cls._is_project_conflict(c_text, e_text)

        return False

    @classmethod
    def _is_fact_conflict(cls, c_text: str, e_text: str) -> bool:
        """Facts conflict if they describe the same attribute slot with different values."""
        for _slot_name, pattern in _FACT_SLOTS:
            if pattern.search(c_text) and pattern.search(e_text):
                return True
        return False

    @classmethod
    def _is_preference_conflict(cls, c_text: str, e_text: str) -> bool:
        """Preferences conflict if they belong to the same mutually exclusive domain or direct contradiction."""
        # 1. Direct contradiction (e.g. likes X vs dislikes X)
        has_negation_c = any(w in c_text for w in ("dislike", "not like", "doesn't like", "hate"))
        has_negation_e = any(w in e_text for w in ("dislike", "not like", "doesn't like", "hate"))
        if has_negation_c != has_negation_e:
            # Check if same subject is involved
            tokens_c = set(re.findall(r"[a-z0-9]+", c_text)) - {"user", "not", "like", "likes", "dislike", "dislikes", "prefers", "prefer"}
            tokens_e = set(re.findall(r"[a-z0-9]+", e_text)) - {"user", "not", "like", "likes", "dislike", "dislikes", "prefers", "prefer"}
            if tokens_c and tokens_c == tokens_e:
                return True

        # 2. Mutually exclusive domain collision (e.g. Python vs JavaScript)
        for cluster in _DOMAIN_CLUSTERS:
            matched_c = [item for item in cluster if item in c_text]
            matched_e = [item for item in cluster if item in e_text]
            if matched_c and matched_e:
                # If they mention different items in the same mutually exclusive cluster, it's a conflict
                if set(matched_c) != set(matched_e):
                    # Check if qualified with a differing sub-domain (e.g. "for dsa" vs "for web")
                    qual_c = cls._extract_qualifier(c_text)
                    qual_e = cls._extract_qualifier(e_text)
                    if qual_c and qual_e and qual_c != qual_e:
                        return False  # "C++ for DSA" does not conflict with "Python for web"
                    return True

        # 3. Canonical predicate match ("User prefers X" vs "User prefers Y" where neither has domain)
        pref_pattern = re.compile(r"^user prefers\s+([^.!?]+)", re.IGNORECASE)
        match_c = pref_pattern.search(c_text)
        match_e = pref_pattern.search(e_text)
        if match_c and match_e:
            val_c = match_c.group(1).strip()
            val_e = match_e.group(1).strip()
            # If both have single-word language/tech values or direct alternative
            if val_c != val_e and (" " not in val_c or " " not in val_e):
                return True

        return False

    @classmethod
    def _is_goal_conflict(cls, c_text: str, e_text: str) -> bool:
        """Goals conflict if they state alternative technologies or goals in the same domain."""
        for cluster in _DOMAIN_CLUSTERS:
            matched_c = [item for item in cluster if item in c_text]
            matched_e = [item for item in cluster if item in e_text]
            if matched_c and matched_e and set(matched_c) != set(matched_e):
                return True

        # "User's goal is to X" vs "User's goal is to Y"
        goal_pattern = re.compile(r"^user(?:'s)?\s+goal\s+is\s+to\s+([^.!?]+)", re.IGNORECASE)
        match_c = goal_pattern.search(c_text)
        match_e = goal_pattern.search(e_text)
        if match_c and match_e and match_c.group(1).strip() != match_e.group(1).strip():
            return True

        return False

    @classmethod
    def _is_project_conflict(cls, c_text: str, e_text: str) -> bool:
        """Projects conflict if they describe the same named project entity."""
        # Extract named project entity (e.g. P-MAI or specific proper nouns)
        # Look for words with hyphens, capitals, or quoted names
        entities_c = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", c_text))
        entities_e = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b", e_text))
        common_entities = entities_c & entities_e
        if common_entities:
            # Same named project entity (e.g. p-mai) with updated state/language
            return True

        # Check "User's main project is X" vs "User's main project is Y"
        main_proj = re.compile(r"main project is\s+([^.!?]+)", re.IGNORECASE)
        m_c = main_proj.search(c_text)
        m_e = main_proj.search(e_text)
        if m_c and m_e and m_c.group(1).strip() != m_e.group(1).strip():
            return True

        return False

    @classmethod
    def _extract_qualifier(cls, text: str) -> str | None:
        """Extract 'for <purpose>' qualifier if present (e.g. 'for dsa', 'for web')."""
        match = re.search(r"\bfor\s+([a-z0-9\s_-]+)$", text)
        return match.group(1).strip() if match else None
