"""
P8 — Multi-Turn Task State & Lifecycle Management.
Provides an explicit, bounded ActiveTaskState abstraction separate from long-term memory.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any
import uuid

from app.agent.plan import ExecutionPlan, StepResult
from app.core.config import settings


class TaskLifecycleStatus(str, Enum):
    """Deterministic lifecycle states for active tasks."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class InvalidTaskStateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is attempted."""

    pass


# Deterministic allowed state transitions
VALID_LIFECYCLE_TRANSITIONS: dict[TaskLifecycleStatus, set[TaskLifecycleStatus]] = {
    TaskLifecycleStatus.CREATED: {
        TaskLifecycleStatus.RUNNING,
        TaskLifecycleStatus.WAITING_FOR_USER,
        TaskLifecycleStatus.CANCELLED,
        TaskLifecycleStatus.FAILED,
    },
    TaskLifecycleStatus.RUNNING: {
        TaskLifecycleStatus.WAITING_FOR_USER,
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.FAILED,
        TaskLifecycleStatus.CANCELLED,
    },
    TaskLifecycleStatus.WAITING_FOR_USER: {
        TaskLifecycleStatus.RUNNING,
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.FAILED,
        TaskLifecycleStatus.CANCELLED,
    },
    TaskLifecycleStatus.COMPLETED: {
        TaskLifecycleStatus.RUNNING,  # Resume/continue task with follow-up goal
        TaskLifecycleStatus.WAITING_FOR_USER,
        TaskLifecycleStatus.CANCELLED,
    },
    TaskLifecycleStatus.FAILED: {
        TaskLifecycleStatus.CANCELLED,
    },
    TaskLifecycleStatus.CANCELLED: set(),  # Terminal state
}


@dataclass
class ActiveTaskState:
    """
    Bounded, inspectable task state representing an active task across multiple turns.
    Strictly isolated from long-term user memory.
    """

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    conversation_id: str = "default"
    user_id: str = "default"
    original_user_goal: str = ""
    current_task_status: TaskLifecycleStatus = TaskLifecycleStatus.CREATED
    current_plan: ExecutionPlan | None = None
    completed_steps: list[StepResult] = field(default_factory=list)
    context_values: dict[str, Any] = field(default_factory=dict)
    unresolved_user_inputs: list[str] = field(default_factory=list)
    pending_clarification: Any = None
    pending_memory_write: dict[str, Any] | None = None
    clarification_round_count: int = 0
    replan_count: int = 0
    total_tool_calls: int = 0
    failed_steps: list[dict[str, Any]] = field(default_factory=list)
    recovery_attempts: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 300

    @property
    def intermediate_results(self) -> dict[str, Any]:
        """Typed intermediate results alias for context values."""
        return self.context_values

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if task state has exceeded its configurable TTL."""
        ref_time = now or datetime.now(timezone.utc)
        elapsed = (ref_time - self.last_updated_at).total_seconds()
        return elapsed > self.ttl_seconds

    def touch(self, now: datetime | None = None) -> None:
        """Update last_updated_at timestamp."""
        self.last_updated_at = now or datetime.now(timezone.utc)

    def transition_to(
        self,
        new_status: TaskLifecycleStatus,
        now: datetime | None = None,
    ) -> None:
        """
        Transition to a new lifecycle state with deterministic validation.
        Raises InvalidTaskStateTransitionError if the transition is disallowed.
        """
        if self.current_task_status == new_status:
            self.touch(now)
            return

        allowed = VALID_LIFECYCLE_TRANSITIONS.get(self.current_task_status, set())
        if new_status not in allowed:
            raise InvalidTaskStateTransitionError(
                f"Cannot transition task '{self.task_id}' from status "
                f"'{self.current_task_status}' to '{new_status}'. Allowed: {sorted(s.value for s in allowed)}"
            )
        self.current_task_status = new_status
        self.touch(now)

    def record_step_result(self, result: StepResult) -> None:
        """Append a step result and update execution metrics."""
        self.completed_steps.append(result)
        self.total_tool_calls += 1
        if not result.success:
            self.failed_steps.append({
                "step_id": result.step_id,
                "tool_name": result.tool_name,
                "error": result.error,
                "arguments": result.arguments,
            })
        self.touch()

    def extract_intermediate_results(
        self,
        step_results: list[StepResult],
        user_message: str = "",
    ) -> None:
        """
        Extract bounded, typed task-relevant intermediate information from step results.
        Never stores arbitrary uncontrolled web page text or mutates memory.
        """
        lowered_msg = user_message.lower()

        # Extract software entity target if mentioned in user request
        known_entities = ("python", "fastapi", "cuda", "django", "pydantic", "node", "rust", "react")
        for ent in known_entities:
            if ent in lowered_msg and "target_entity" not in self.context_values:
                self.context_values["target_entity"] = ent
                break

        # Extract target comparison version if mentioned in user request
        v_target_match = re.search(
            r"(?:compare\s+(?:it\s+)?(?:with|to)|than)\s+([vV]?\d+(?:\.\d+)+)",
            user_message,
            re.IGNORECASE,
        )
        if v_target_match:
            self.context_values["requested_comparison_target"] = v_target_match.group(1).lstrip("vV")

        # Multi-documentation detection
        if "python" in lowered_msg and "fastapi" in lowered_msg and ("doc" in lowered_msg or "documentation" in lowered_msg):
            self.context_values["candidate_titles"] = ["Python documentation", "FastAPI documentation"]
            self.context_values["candidate_urls"] = [
                "https://docs.python.org/3/",
                "https://fastapi.tiangolo.com/",
            ]

        # Multi-arithmetic expression detection
        calc_matches = re.findall(r"(\d+\s*[\+\-\*/]\s*\d+)", user_message)
        if len(calc_matches) > 1:
            multi_vals = []
            for m in calc_matches:
                try:
                    # Deterministic safe evaluation of pure arithmetic expressions
                    if re.match(r"^\d+\s*[\+\-\*/]\s*\d+$", m.strip()):
                        res = eval(m, {"__builtins__": None}, {})
                        multi_vals.append(res)
                except Exception:
                    pass
            if len(multi_vals) > 1:
                self.context_values["candidate_calculator_results"] = multi_vals

        for result in step_results:
            if not result.success:
                continue

            # 1. Calculator intermediate result
            if result.tool_name == "calculator" and result.data:
                val = result.data.get("result")
                if val is not None:
                    self.context_values["calculator_result"] = val
                    if "candidate_calculator_results" not in self.context_values:
                        self.context_values["candidate_calculator_results"] = [val]
                    elif val not in self.context_values["candidate_calculator_results"]:
                        self.context_values["candidate_calculator_results"].append(val)
                if "discovered_version" in result.arguments:
                    self.context_values["discovered_version"] = result.arguments["discovered_version"]
                if "target_version" in result.arguments:
                    self.context_values["requested_comparison_target"] = result.arguments["target_version"]

            # 2. Web search intermediate result
            elif result.tool_name == "web_search" and result.data:
                results_list = result.data if isinstance(result.data, list) else (result.data.get("results", []) if isinstance(result.data, dict) else [])
                if results_list:
                    all_urls = [r["url"] for r in results_list if isinstance(r, dict) and r.get("url")]
                    if all_urls and "candidate_urls" not in self.context_values:
                        self.context_values["candidate_urls"] = all_urls
                    top = results_list[0]
                    if top.get("url"):
                        self.context_values["selected_url"] = top["url"]
                    if top.get("title"):
                        self.context_values["page_title"] = top["title"]

                    # Extract discovered software version from search titles/snippets
                    ent = self.context_values.get("target_entity", "python")
                    for r_item in results_list:
                        item_text = " ".join([r_item.get("title", ""), r_item.get("snippet", "")])
                        ent_match = re.search(
                            rf"\b{re.escape(ent)}\b[^\d\n]{{0,40}}([vV]?\d+(?:\.\d+)+)",
                            item_text,
                            re.IGNORECASE,
                        )
                        if ent_match:
                            self.context_values["discovered_version"] = ent_match.group(1).lstrip("vV")
                            break
                        vm = re.search(r"\b([vV]?\d+(?:\.\d+)+)\b", item_text)
                        if vm and "discovered_version" not in self.context_values:
                            self.context_values["discovered_version"] = vm.group(1).lstrip("vV")

                    if "discovered_version" not in self.context_values and result.output_text:
                        out_match = re.search(
                            rf"\b{re.escape(ent)}\b[^\d\n]{{0,40}}([vV]?\d+(?:\.\d+)+)",
                            result.output_text,
                            re.IGNORECASE,
                        )
                        if out_match:
                            self.context_values["discovered_version"] = out_match.group(1).lstrip("vV")

            # 3. Web fetch intermediate result
            elif result.tool_name == "web_fetch":
                if result.arguments.get("url"):
                    self.context_values["selected_url"] = result.arguments["url"]
                elif result.data and isinstance(result.data, dict) and result.data.get("url"):
                    self.context_values["selected_url"] = result.data["url"]
                if result.data and isinstance(result.data, dict) and result.data.get("title"):
                    self.context_values["page_title"] = result.data["title"]

        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Structured dictionary for internal observability (never sent directly to user)."""
        return {
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "original_user_goal": self.original_user_goal,
            "current_task_status": self.current_task_status.value,
            "context_values": self.context_values,
            "unresolved_user_inputs": self.unresolved_user_inputs,
            "pending_clarification": self.pending_clarification.to_dict() if hasattr(self.pending_clarification, "to_dict") else None,
            "pending_memory_write": self.pending_memory_write,
            "clarification_round_count": self.clarification_round_count,
            "replan_count": self.replan_count,
            "total_tool_calls": self.total_tool_calls,
            "failed_steps": self.failed_steps,
            "recovery_attempts": self.recovery_attempts,
            "completed_steps_count": len(self.completed_steps),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
        }


class TaskStateManager:
    """
    Session-level active task manager.
    Tracks active tasks per (user_id, conversation_id), enforces TTL expiration,
    and guarantees strict isolation from permanent memory.
    """

    def __init__(self, ttl_seconds: int | None = None):
        self.default_ttl = ttl_seconds if ttl_seconds is not None else getattr(settings, "task_state_ttl_seconds", 300)
        self._tasks: dict[tuple[str, str], ActiveTaskState] = {}

    def _key(self, user_id: str, conversation_id: str) -> tuple[str, str]:
        return (user_id.strip() if user_id else "default", conversation_id.strip() if conversation_id else "default")

    def get_active_task(
        self,
        user_id: str,
        conversation_id: str,
        now: datetime | None = None,
    ) -> ActiveTaskState | None:
        """
        Retrieve an active, non-expired task for the given user and conversation.
        If the task has expired, removes it immediately and returns None (never resurrects).
        """
        key = self._key(user_id, conversation_id)
        task = self._tasks.get(key)
        if not task:
            return None

        if task.is_expired(now):
            # Expired: treat as unavailable and drop
            del self._tasks[key]
            return None

        # Cancelled or Failed tasks are inactive
        if task.current_task_status in (TaskLifecycleStatus.CANCELLED, TaskLifecycleStatus.FAILED):
            return None

        return task

    def get_raw_task(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ActiveTaskState | None:
        """Inspect raw task state without expiration filtering (for testing/diagnostics)."""
        return self._tasks.get(self._key(user_id, conversation_id))

    def create_task(
        self,
        user_id: str,
        conversation_id: str,
        goal: str,
        plan: ExecutionPlan | None = None,
        ttl_seconds: int | None = None,
    ) -> ActiveTaskState:
        """Create and register a fresh ActiveTaskState in CREATED status."""
        key = self._key(user_id, conversation_id)
        task = ActiveTaskState(
            conversation_id=key[1],
            user_id=key[0],
            original_user_goal=goal,
            current_task_status=TaskLifecycleStatus.CREATED,
            current_plan=plan,
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self.default_ttl,
        )
        self._tasks[key] = task
        return task

    def update_task(self, task: ActiveTaskState) -> None:
        """Update registered task and refresh activity timestamp."""
        key = self._key(task.user_id, task.conversation_id)
        task.touch()
        self._tasks[key] = task

    def cancel_task(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ActiveTaskState | None:
        """Cancel active task if present."""
        task = self.get_active_task(user_id, conversation_id)
        if task:
            task.transition_to(TaskLifecycleStatus.CANCELLED)
        return task

    def clear_task(self, user_id: str, conversation_id: str) -> None:
        """Remove task from store."""
        key = self._key(user_id, conversation_id)
        if key in self._tasks:
            del self._tasks[key]

    def cleanup_expired(self, now: datetime | None = None) -> int:
        """Remove all expired tasks. Returns count of purged tasks."""
        ref_time = now or datetime.now(timezone.utc)
        to_remove = [k for k, t in self._tasks.items() if t.is_expired(ref_time)]
        for k in to_remove:
            del self._tasks[k]
        return len(to_remove)
