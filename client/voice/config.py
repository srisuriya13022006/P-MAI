"""
P15 — Voice Client Configuration.
Holds user-configurable client parameters with strict security invariants:
Never stores or exposes server-side credentials or API keys.
"""
from pydantic import BaseModel, Field


class VoiceClientConfig(BaseModel):
    """Configuration settings for P-MAI Voice Client."""

    backend_http_url: str = Field(
        default="http://localhost:8000",
        description="P-MAI backend HTTP base URL",
    )
    backend_ws_url: str = Field(
        default="ws://localhost:8000/voice/realtime",
        description="P-MAI real-time voice WebSocket endpoint URL",
    )
    conversation_id: str = Field(
        default="default",
        description="Conversation session identifier",
    )
    user_id: str = Field(
        default="default",
        description="User identifier",
    )
    max_reconnect_attempts: int = Field(
        default=3,
        description="Maximum consecutive reconnection attempts before giving up",
    )
    reconnect_delay_seconds: float = Field(
        default=1.0,
        description="Base delay between reconnection attempts in seconds",
    )
    audio_chunk_duration_s: float = Field(
        default=0.2,
        description="Audio slice duration in seconds for streaming capture",
    )
    max_audio_buffer_bytes: int = Field(
        default=5 * 1024 * 1024,
        description="Maximum client audio memory buffer (5 MB)",
    )
    max_chunk_size_bytes: int = Field(
        default=64 * 1024,
        description="Maximum single audio chunk size (64 KB)",
    )
    max_history_turns: int = Field(
        default=20,
        description="Maximum conversational turns displayed in client history",
    )
    sample_rate: int = Field(
        default=16000,
        description="Canonical audio sample rate (16 kHz)",
    )
    channels: int = Field(
        default=1,
        description="Canonical audio channels (Mono)",
    )
