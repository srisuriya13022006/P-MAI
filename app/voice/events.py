"""
P12.2 — Real-time Voice Interaction Event Protocol.
Deterministic event definitions for streaming audio, transcripts, TTS, and barge-in.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class RealtimeEventType(str, Enum):
    """Deterministic streaming voice event types."""

    # Audio transport
    AUDIO_START = "AUDIO_START"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    AUDIO_END = "AUDIO_END"

    # Voice activity detection
    SPEECH_START = "SPEECH_START"
    SPEECH_END = "SPEECH_END"

    # Speech to text
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"

    # Grounded response & Text to speech streaming
    RESPONSE_START = "RESPONSE_START"
    RESPONSE_CHUNK = "RESPONSE_CHUNK"
    RESPONSE_END = "RESPONSE_END"

    # Barge-in / Interruption
    INTERRUPT = "INTERRUPT"

    # Session control & status
    ERROR = "ERROR"
    CLOSE = "CLOSE"


class RealtimeClientEvent(BaseModel):
    """Event sent from client to server."""

    type: RealtimeEventType = Field(..., description="Event type")
    session_id: str | None = Field(default=None, description="Active session ID")
    audio_base64: str | None = Field(default=None, description="Base64 encoded audio fragment")
    content_type: str | None = Field(default="audio/wav", description="Audio format")
    reason: str | None = Field(default=None, description="Reason for event (e.g. interruption, close)")
    payload: dict[str, Any] = Field(default_factory=dict, description="Additional client payload")


class RealtimeServerEvent(BaseModel):
    """Event emitted from server to client."""

    type: RealtimeEventType = Field(..., description="Event type")
    session_id: str = Field(..., description="Active session ID")
    turn_index: int = Field(default=0, description="Current conversational turn")
    transcript: str | None = Field(default=None, description="Transcribed partial or final text")
    text_chunk: str | None = Field(default=None, description="Incremental text fragment")
    audio_base64: str | None = Field(default=None, description="Incremental synthesized audio chunk")
    error_code: str | None = Field(default=None, description="Structured error code if error occurred")
    error_message: str | None = Field(default=None, description="User-facing error description")
    is_interrupted: bool = Field(default=False, description="True if response was cut short by barge-in")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Timing / observability metrics")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
