"""
P12 — Voice I/O Configuration.
Provides structured, configurable settings for voice processing with safe defaults.
"""
from dataclasses import dataclass, field
from typing import Any


DEFAULT_SUPPORTED_AUDIO_FORMATS = [
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
]


@dataclass
class VoiceConfig:
    """Configurable parameters for voice I/O pipeline."""

    enabled: bool = True
    stt_provider: str = "assemblyai"
    tts_provider: str = "local"
    supported_formats: list[str] = field(default_factory=lambda: list(DEFAULT_SUPPORTED_AUDIO_FORMATS))
    max_audio_size_bytes: int = 10 * 1024 * 1024  # 10 MB limit
    max_duration_seconds: float = 60.0  # 60s max audio duration
    timeout_seconds: float = 15.0  # Timeout for provider requests
    voice_timeout_seconds: float | None = None
    default_language: str = "en"
    default_voice: str = "natural-en-1"
    tts_voice: str | None = None
    tts_audio_format: str = "audio/wav"
    vad_silence_threshold_db: float = -40.0
    vad_max_silence_seconds: float = 2.0
    # Realtime Voice parameters
    realtime_enabled: bool = True
    realtime_max_session_duration_seconds: float = 300.0
    realtime_inactivity_timeout_seconds: float = 60.0
    realtime_max_utterance_duration_seconds: float = 30.0
    realtime_max_concurrent_sessions: int = 10
    realtime_max_audio_buffer_bytes: int = 2 * 1024 * 1024  # 2MB backpressure limit
    realtime_tts_chunk_duration_seconds: float = 0.5
    realtime_stop_phrases: list[str] = field(
        default_factory=lambda: ["stop", "stop talking", "be quiet", "pause", "shut up", "hold on"]
    )
    realtime_cancel_phrases: list[str] = field(
        default_factory=lambda: ["cancel this task", "cancel task", "stop this task", "abort task", "cancel"]
    )
    # Production Provider Settings (P13 / P15.4)
    stt_api_key: str | None = None
    tts_api_key: str | None = None
    stt_base_url: str = "https://api.groq.com/openai/v1"
    tts_base_url: str = "https://api.openai.com/v1"
    sample_rate: int = 16000
    audio_format: str = "audio/wav"
    # AssemblyAI Provider Settings (P15.4)
    assemblyai_api_key: str | None = None
    assemblyai_stt_model: str = "universal-3-6-pro"
    stt_fallback_enabled: bool = True

    # Local-First Provider Settings (P14)
    stt_local_model: str = "base"
    tts_local_engine: str = "pyttsx3"

    def __post_init__(self):
        if self.voice_timeout_seconds is not None:
            self.timeout_seconds = self.voice_timeout_seconds
        if self.tts_voice is not None:
            self.default_voice = self.tts_voice
        else:
            self.tts_voice = self.default_voice

    @classmethod
    def from_settings(cls, settings_obj: Any = None) -> "VoiceConfig":
        """Construct VoiceConfig pulling from application settings if available."""
        if settings_obj is None:
            from app.core.config import settings
            settings_obj = settings

        return cls(
            enabled=getattr(settings_obj, "voice_enabled", True),
            stt_provider=getattr(settings_obj, "voice_stt_provider", "assemblyai"),
            tts_provider=getattr(settings_obj, "voice_tts_provider", "local"),
            supported_formats=getattr(settings_obj, "voice_supported_formats", DEFAULT_SUPPORTED_AUDIO_FORMATS),
            max_audio_size_bytes=getattr(settings_obj, "voice_max_audio_size_bytes", 10 * 1024 * 1024),
            max_duration_seconds=getattr(settings_obj, "voice_max_duration_seconds", 60.0),
            timeout_seconds=getattr(settings_obj, "voice_timeout_seconds", 15.0),
            default_language=getattr(settings_obj, "voice_default_language", "en"),
            default_voice=getattr(settings_obj, "voice_default_tts_voice", "natural-en-1"),
            vad_silence_threshold_db=getattr(settings_obj, "voice_vad_silence_threshold_db", -40.0),
            vad_max_silence_seconds=getattr(settings_obj, "voice_vad_max_silence_seconds", 2.0),
            stt_api_key=getattr(settings_obj, "voice_stt_api_key", None),
            tts_api_key=getattr(settings_obj, "voice_tts_api_key", None),
            stt_base_url=getattr(settings_obj, "voice_stt_base_url", "https://api.groq.com/openai/v1"),
            tts_base_url=getattr(settings_obj, "voice_tts_base_url", "https://api.openai.com/v1"),
            sample_rate=getattr(settings_obj, "voice_sample_rate", 16000),
            audio_format=getattr(settings_obj, "voice_audio_format", "audio/wav"),
            assemblyai_api_key=getattr(settings_obj, "assemblyai_api_key", None),
            assemblyai_stt_model=getattr(settings_obj, "assemblyai_stt_model", "universal-3-6-pro"),

            stt_fallback_enabled=getattr(settings_obj, "stt_fallback_enabled", True),
            stt_local_model=getattr(settings_obj, "voice_stt_local_model", "base"),
            tts_local_engine=getattr(settings_obj, "voice_tts_local_engine", "pyttsx3"),
        )
