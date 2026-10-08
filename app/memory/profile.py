"""User profile context builder for cross-session memory continuity.

Selects high-value, active, non-expired memories to construct a concise
'USER PROFILE' context injected at the start of new conversations.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING

from app.memory.temporal import calculate_temporal_score, is_expired

if TYPE_CHECKING:
    from app.database.models import Memory

logger = logging.getLogger(__name__)

# Configurable profile budget
DEFAULT_MAX_PROFILE_MEMORIES: int = 5
DEFAULT_MAX_PROFILE_CHARACTERS: int = 1000

# Priority weights for memory types in profile selection
TYPE_PROFILE_WEIGHT: dict[str, float] = {
    "fact": 1.2,       # Core identity (name, location, occupation)
    "project": 1.1,    # Major ongoing projects
    "goal": 1.0,       # Current learning/career targets
    "preference": 0.9, # Coding/tool preferences
}


class MemoryProfileBuilder:
    """Selects and formats persistent active memories into a compact user profile."""

    def __init__(
        self,
        max_memories: int = DEFAULT_MAX_PROFILE_MEMORIES,
        max_characters: int = DEFAULT_MAX_PROFILE_CHARACTERS,
    ) -> None:
        self.max_memories = max_memories
        self.max_characters = max_characters

    def select_memories(
        self,
        memories: list[Memory],
        now: datetime | None = None,
    ) -> list[Memory]:
        """Rank and select a bounded list of high-value active memories for the profile."""
        current_time = now or datetime.utcnow()
        candidates: list[tuple[float, int, datetime, Memory]] = []

        for memory in memories:
            # 1. Respect active state
            if not getattr(memory, "is_active", True):
                continue

            # 2. Skip expired memories
            if is_expired(memory, current_time):
                continue

            # 3. Compute profile score using importance, type priority, and temporal freshness
            normalized_importance = max(1.0, min(3.0, float(memory.importance))) / 3.0
            type_weight = TYPE_PROFILE_WEIGHT.get(memory.memory_type, 1.0)
            temporal_score = calculate_temporal_score(memory, now=current_time)

            # High importance + identity types + fresh/permanent memories score highest
            profile_score = normalized_importance * type_weight * temporal_score
            timestamp = memory.updated_at or memory.created_at or current_time

            candidates.append((profile_score, memory.importance, timestamp, memory))

        # Sort descending by composite score, tie-breaking by importance and timestamp
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

        # Apply memory count limit
        selected: list[Memory] = []
        total_chars = 0

        for _score, _importance, _ts, memory in candidates[: self.max_memories]:
            content_len = len(memory.content) + 4  # bullet + spacing
            if total_chars + content_len > self.max_characters and selected:
                break
            selected.append(memory)
            total_chars += content_len

        logger.debug(
            "Selected %d profile memories (from %d active candidates)",
            len(selected),
            len(memories),
        )
        return selected

    def format_profile(self, memories: list[Memory]) -> str:
        """Format selected memories into a concise text block."""
        if not memories:
            logger.debug("Profile skipped: no memories to include")
            return ""

        bullet_points = []
        for mem in memories:
            text = mem.content.strip()
            if not text.endswith((".", "!", "?")):
                text += "."
            bullet_points.append(f"- {text}")

        lines = "\n".join(bullet_points)
        return f"USER PROFILE:\n{lines}"

    def build(
        self,
        memories: list[Memory],
        now: datetime | None = None,
    ) -> str:
        """Convenience method to select and format memories in one call."""
        selected = self.select_memories(memories, now=now)
        return self.format_profile(selected)
