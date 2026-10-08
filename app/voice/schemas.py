"""
P12 — Voice I/O Schemas.
Pydantic models and dataclasses for voice input, output, STT/TTS results, and timing.
"""
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, Field


@dataclass
class AudioValidationResult:
    """Result of audio format and constraint validation."""

    is_valid: bool
    error_message: str | None = None
    error_code: str | None = None
    content_type: str | None = None
    size_bytes: int = 0
    estimated_duration_seconds: float | None = None


@dataclass
class STTResult:
    """Output from a Speech-to-Text provider."""

    transcript: str
    language: str | None = "en"
    duration_seconds: float | None = None
    provider: str = "mock"
    confidence: float | None = 1.0


@dataclass
class TTSResult:
    """Output from a Text-to-Speech provider."""

    audio_bytes: bytes
    content_type: str = "audio/wav"
    duration_seconds: float | None = None
    provider: str = "mock"


class VoiceTimingMetadata(BaseModel):
    """Execution timing metadata for observability (never exposes internal agent IDs)."""

    stt_latency_ms: float = Field(default=0.0, description="Latency of speech-to-text transcription")
    mai_latency_ms: float = Field(default=0.0, description="Latency of MAI brain processing and reasoning")
    tts_latency_ms: float = Field(default=0.0, description="Latency of text-to-speech synthesis")
    total_latency_ms: float = Field(default=0.0, description="End-to-end voice turnaround time")


class VoiceChatRequest(BaseModel):
    """Voice chat request payload accepting base64-encoded audio."""

    user_id: str = Field(default="default", description="User ID for profile/context scoping")
    conversation_id: str = Field(default="default", description="Session conversation ID")
    audio_base64: str = Field(..., description="Base64 encoded audio byte stream")
    content_type: str = Field(default="audio/wav", description="MIME content type of audio")
    language: str | None = Field(default=None, description="Optional spoken language code")


class VoiceChatResponse(BaseModel):
    """
    Clean voice response contract.
    Never exposes internal task IDs, trace IDs, planner metadata, or recovery details.
    """

    conversation_id: str = Field(..., description="Conversation ID")
    transcript: str = Field(..., description="Transcribed user speech")
    text_response: str = Field(..., description="Final grounded text answer from MAI brain")
    audio_base64: str | None = Field(default=None, description="Base64 synthesized audio response")
    audio_content_type: str | None = Field(default="audio/wav", description="Audio MIME format")
    status: str = Field(default="success", description="Overall execution status: success, partial_success, error")
    timing: VoiceTimingMetadata | None = Field(default=None, description="Timing observability metadata")
    tts_error: str | None = Field(default=None, description="Non-fatal TTS failure message if audio synthesis failed")
