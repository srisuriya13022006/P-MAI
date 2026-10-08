"""Deterministic temporal scoring and decay for MAI memories.

Rationale & Mathematical Formula:
---------------------------------
Memories should not disappear purely because time has passed; instead, newer
information should receive greater temporal relevance when competing against
similar memories. Crucially, permanent identity facts (e.g. "User's name is Suriya")
must be protected from aggressive decay.

Formula:
  1. Expiration check:
       If memory.expires_at is present and now > memory.expires_at:
           Score = 0.0 (expired).

  2. Age Calculation:
       timestamp = memory.updated_at or memory.created_at
       age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
       If timestamp is None: fail-safe return 1.0.

  3. Effective Half-Life (H):
       H = DEFAULT_HALF_LIFE_DAYS * TYPE_HALF_LIFE_MULTIPLIER[memory_type] * (importance / 2.0)
       - "goal": ~30 days (fast evolution)
       - "project": ~60 days
       - "preference": ~90 days
       - "fact": ~360 days (permanent / long-term)

  4. Exponential Decay:
       decay = 0.5 ** (age_days / H)

  5. Floor Protection:
       floor = FACT_OR_HIGH_IMPORTANCE_FLOOR (0.80) if importance >= 3 or memory_type == 'fact'
               else DEFAULT_DECAY_FLOOR (0.20)
       temporal_score = max(floor, decay)

This ensures:
  - Brand new memories have score 1.0.
  - Facts & high-importance memories never decay below 0.80 regardless of age.
  - Goals decay noticeably within 1-2 months, allowing newer goals to be ranked higher.
  - Zero network/LLM dependencies; 100% deterministic and testable.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database.models import Memory

# Configurable temporal parameters
DEFAULT_HALF_LIFE_DAYS: float = 30.0

TYPE_HALF_LIFE_MULTIPLIER: dict[str, float] = {
    "fact": 12.0,       # 360 days base half-life
    "preference": 3.0,  # 90 days base half-life
    "project": 2.0,     # 60 days base half-life
    "goal": 1.0,        # 30 days base half-life
}

FACT_OR_HIGH_IMPORTANCE_FLOOR: float = 0.80
DEFAULT_DECAY_FLOOR: float = 0.20


def is_expired(memory: Memory, now: datetime | None = None) -> bool:
    """Return True if the memory has an explicit expiration timestamp that has passed."""
    if memory.expires_at is None:
        return False
    current_time = now or datetime.utcnow()
    return current_time > memory.expires_at


def calculate_temporal_score(
    memory: Memory,
    now: datetime | None = None,
    default_half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Calculate deterministic temporal freshness score in [0.0, 1.0].

    Fails safely and returns 1.0 if memory has no timestamp.
    Returns 0.0 if the memory is expired.
    """
    if is_expired(memory, now):
        return 0.0

    timestamp = memory.updated_at or memory.created_at
    if timestamp is None:
        return 1.0

    current_time = now or datetime.utcnow()
    age_seconds = (current_time - timestamp).total_seconds()
    age_days = max(0.0, age_seconds / 86400.0)

    # Calculate type- and importance-adjusted half life
    type_mult = TYPE_HALF_LIFE_MULTIPLIER.get(memory.memory_type, 1.0)
    importance_factor = max(0.5, float(memory.importance) / 2.0)
    effective_half_life = max(1.0, default_half_life_days * type_mult * importance_factor)

    decay = 0.5 ** (age_days / effective_half_life)

    # Floor protection for facts and high-importance memories
    is_protected = memory.memory_type == "fact" or memory.importance >= 3
    floor = FACT_OR_HIGH_IMPORTANCE_FLOOR if is_protected else DEFAULT_DECAY_FLOOR

    return float(max(floor, min(1.0, decay)))
