"""
P12.2 — Real-time Voice Session Management.
Maintains session state, deterministic lifecycle transitions, bounds, and concurrency control.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import time
import uuid
from typing import Any

from app.voice.config import VoiceConfig
from app.voice.exceptions import (
    BackpressureError,
    InvalidSessionStateTransitionError,
    SessionClosedError,
    SessionExpiredError,
    SessionInactivityTimeoutError,
    SessionLimitExceededError,
    UnauthorizedSessionError,
)


class SessionStatus(str, Enum):
    """Deterministic lifecycle states for real-time voice sessions."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    CLOSED = "CLOSED"


VALID_SESSION_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.IDLE: {
        SessionStatus.LISTENING,
        SessionStatus.CLOSED,
    },
    SessionStatus.LISTENING: {
        SessionStatus.TRANSCRIBING,
        SessionStatus.INTERRUPTED,
        SessionStatus.IDLE,
        SessionStatus.CLOSED,
    },
    SessionStatus.TRANSCRIBING: {
        SessionStatus.PROCESSING,
        SessionStatus.LISTENING,
        SessionStatus.INTERRUPTED,
        SessionStatus.IDLE,
        SessionStatus.CLOSED,
    },
    SessionStatus.PROCESSING: {
        SessionStatus.SPEAKING,
        SessionStatus.INTERRUPTED,
        SessionStatus.IDLE,
        SessionStatus.CLOSED,
    },
    SessionStatus.SPEAKING: {
        SessionStatus.INTERRUPTED,
        SessionStatus.LISTENING,
        SessionStatus.IDLE,
        SessionStatus.CLOSED,
    },
    SessionStatus.INTERRUPTED: {
        SessionStatus.LISTENING,
        SessionStatus.TRANSCRIBING,
        SessionStatus.IDLE,
        SessionStatus.CLOSED,
    },
    SessionStatus.CLOSED: set(),
}


@dataclass
class RealtimeVoiceSession:
    """Represents a bounded real-time conversational voice session."""

    session_id: str
    conversation_id: str
    user_id: str
    config: VoiceConfig
    status: SessionStatus = SessionStatus.IDLE
    current_turn: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)

    # State tracking
    is_speech_active: bool = False
    is_interrupted: bool = False
    interruption_count: int = 0
    current_stt_transcript: str = ""
    current_response_text: str = ""

    # Streaming audio buffer
    audio_buffer: bytearray = field(default_factory=bytearray)
    total_audio_received_bytes: int = 0

    def transition_to(self, new_status: SessionStatus) -> None:
        """Enforce strict, deterministic state machine transitions."""
        if self.status == SessionStatus.CLOSED:
            raise SessionClosedError(self.session_id)

        allowed = VALID_SESSION_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidSessionStateTransitionError(self.status.value, new_status.value)

        self.status = new_status
        self.touch()

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = time.time()

    def check_bounds(self) -> None:
        """Validate lifetime and inactivity bounds."""
        if self.status == SessionStatus.CLOSED:
            return

        now = time.time()
        elapsed = now - self.created_at
        if elapsed > self.config.realtime_max_session_duration_seconds:
            self.status = SessionStatus.CLOSED
            raise SessionExpiredError(elapsed, self.config.realtime_max_session_duration_seconds)

        idle = now - self.last_activity_at
        if idle > self.config.realtime_inactivity_timeout_seconds:
            self.status = SessionStatus.CLOSED
            raise SessionInactivityTimeoutError(idle, self.config.realtime_inactivity_timeout_seconds)

    def append_audio_chunk(self, chunk: bytes) -> None:
        """Append incoming audio bytes with strict backpressure validation."""
        self.touch()
        if len(self.audio_buffer) + len(chunk) > self.config.realtime_max_audio_buffer_bytes:
            raise BackpressureError(
                len(self.audio_buffer) + len(chunk),
                self.config.realtime_max_audio_buffer_bytes,
            )
        self.audio_buffer.extend(chunk)
        self.total_audio_received_bytes += len(chunk)

    def clear_audio_buffer(self) -> bytes:
        """Extract and clear current audio buffer."""
        data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return data


class VoiceSessionManager:
    """Manages active real-time voice sessions with concurrency and lifetime limits."""

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self._sessions: dict[str, RealtimeVoiceSession] = {}
        self._lock = threading.Lock()

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.status != SessionStatus.CLOSED)

    def create_session(
        self,
        conversation_id: str = "default",
        user_id: str = "default",
        session_id: str | None = None,
    ) -> RealtimeVoiceSession:
        """Create a new bounded voice session enforcing concurrency limit."""
        with self._lock:
            # Clean up closed or expired sessions first
            self._cleanup_expired_locked()

            active = sum(1 for s in self._sessions.values() if s.status != SessionStatus.CLOSED)
            if active >= self.config.realtime_max_concurrent_sessions:
                raise SessionLimitExceededError(active, self.config.realtime_max_concurrent_sessions)

            s_id = session_id or str(uuid.uuid4())
            session = RealtimeVoiceSession(
                session_id=s_id,
                conversation_id=conversation_id,
                user_id=user_id,
                config=self.config,
                status=SessionStatus.IDLE,
            )
            self._sessions[s_id] = session
            return session

    def get_session(self, session_id: str) -> RealtimeVoiceSession:
        """Retrieve session and validate bounds."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise UnauthorizedSessionError(f"Session '{session_id}' not found.")
            session.check_bounds()
            return session

    def close_session(self, session_id: str) -> None:
        """Close session explicitly."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status != SessionStatus.CLOSED:
                session.status = SessionStatus.CLOSED

    def _cleanup_expired_locked(self) -> None:
        """Internal sweep for expired sessions."""
        now = time.time()
        to_close = []
        for s_id, s in self._sessions.items():
            if s.status != SessionStatus.CLOSED:
                if (
                    now - s.created_at > self.config.realtime_max_session_duration_seconds
                    or now - s.last_activity_at > self.config.realtime_inactivity_timeout_seconds
                ):
                    to_close.append(s_id)
        for s_id in to_close:
            self._sessions[s_id].status = SessionStatus.CLOSED
