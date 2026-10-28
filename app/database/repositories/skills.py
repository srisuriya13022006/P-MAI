"""
P16 — Skill Repository for Procedural Memory.
Handles database operations for saved procedural skills, scoped strictly by user_id.
"""
from datetime import datetime, timezone
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Skill


class SkillRepository:
    """Database operations for durable procedural skills."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        user_id: str,
        name: str,
        description: str,
        steps: list[dict[str, Any]],
        parameters_schema: dict[str, Any] | None = None,
        intended_use: str | None = None,
        completion_criteria: str | None = None,
        provenance: str = "derived_from_task",
        skill_id: str | None = None,
    ) -> Skill:
        """Create and persist a new approved skill."""
        s_id = skill_id or f"skill_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        skill = Skill(
            id=s_id,
            user_id=user_id,
            name=name.strip(),
            description=description.strip(),
            intended_use=intended_use.strip() if intended_use else None,
            parameters_schema=parameters_schema or {},
            steps=steps,
            completion_criteria=completion_criteria,
            is_active=True,
            version=1,
            provenance=provenance,
            usage_count=0,
            last_used_at=None,
            recent_outcomes=[],
            needs_repair=False,
            created_at=now,
            updated_at=now,
        )
        self.db.add(skill)
        self.db.flush()
        return skill

    def get(self, user_id: str, skill_id: str) -> Skill | None:
        """Retrieve a skill by its unique ID for a specific user."""
        statement = select(Skill).where(
            Skill.user_id == user_id,
            Skill.id == skill_id,
        )
        return self.db.scalar(statement)

    def get_by_name(self, user_id: str, name: str, active_only: bool = True) -> Skill | None:
        """Retrieve a skill by name (case-insensitive) for a specific user."""
        clean_name = name.strip().lower()
        conditions = [
            Skill.user_id == user_id,
            func.lower(Skill.name) == clean_name,
        ]
        if active_only:
            conditions.append(Skill.is_active.is_(True))
        statement = select(Skill).where(*conditions)
        return self.db.scalar(statement)

    def list_for_user(self, user_id: str, active_only: bool = True) -> list[Skill]:
        """List all skills owned by the user, ordered by usage and last updated."""
        conditions = [Skill.user_id == user_id]
        if active_only:
            conditions.append(Skill.is_active.is_(True))
        statement = (
            select(Skill)
            .where(*conditions)
            .order_by(Skill.usage_count.desc(), Skill.updated_at.desc())
        )
        return list(self.db.scalars(statement).all())

    def update(self, skill: Skill) -> Skill:
        """Save updates to an existing skill and touch updated_at."""
        skill.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return skill

    def delete(self, user_id: str, skill_id: str) -> bool:
        """Delete a skill belonging to the user."""
        skill = self.get(user_id, skill_id)
        if not skill:
            return False
        self.db.delete(skill)
        self.db.flush()
        return True

    def record_outcome(
        self,
        user_id: str,
        skill_id: str,
        success: bool,
        error: str | None = None,
        steps_executed: int = 0,
        duration_ms: float = 0.0,
    ) -> Skill | None:
        """Record execution outcome, increment usage count, and update last_used_at."""
        skill = self.get(user_id, skill_id)
        if not skill:
            return None

        now = datetime.now(timezone.utc)
        outcome = {
            "timestamp": now.isoformat(),
            "success": success,
            "error": error,
            "steps_executed": steps_executed,
            "duration_ms": round(duration_ms, 2),
        }
        # Keep last 10 outcomes
        outcomes = list(skill.recent_outcomes or [])
        outcomes.append(outcome)
        if len(outcomes) > 10:
            outcomes = outcomes[-10:]

        skill.recent_outcomes = outcomes
        skill.usage_count += 1
        skill.last_used_at = now
        skill.updated_at = now
        self.db.flush()
        return skill
