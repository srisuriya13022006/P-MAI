"""
P13 / P14 / P15.4 — Speech-to-Text Provider Factory.
Instantiates configured cloud (AssemblyAI primary), local (Whisper fallback), or mock STT adapters.
"""
from app.voice.config import VoiceConfig
from app.voice.exceptions import STTUnavailableError
from app.voice.stt.assemblyai_adapter import AssemblyAIRealtimeSTTAdapter
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.stt.mock import MockSTTProvider
from app.voice.stt.resilient_adapter import ResilientSTTAdapter
from app.voice.stt.whisper_adapter import WhisperSTTAdapter


def create_stt_provider(
    config: VoiceConfig | None = None,
    allow_fallback: bool | None = None,
) -> SpeechToTextProvider:
    """
    Create configured STT provider instance.
    Supports 'assemblyai' (P15.4 primary), 'local', 'whisper', 'groq', 'openai', and 'mock'.
    """
    cfg = config or VoiceConfig.from_settings()
    name = (cfg.stt_provider or "").lower().strip()
    fallback_enabled = (
        allow_fallback if allow_fallback is not None else getattr(cfg, "stt_fallback_enabled", True)
    )

    if name == "mock":
        return MockSTTProvider()

    elif name in ("assemblyai", "aai", "default", ""):
        primary = AssemblyAIRealtimeSTTAdapter(
            api_key=cfg.assemblyai_api_key or cfg.stt_api_key,
            model_name=cfg.assemblyai_stt_model,
            timeout_seconds=cfg.timeout_seconds,
            sample_rate=cfg.sample_rate,
            language=cfg.default_language,
        )

        if not fallback_enabled:
            # When fallback is disabled, missing API key produces deterministic configuration error immediately
            primary._require_api_key()
            return primary

        fallback = LocalWhisperSTTAdapter(
            model_name=cfg.stt_local_model,
            timeout_seconds=cfg.timeout_seconds,
            language=cfg.default_language,
        )

        return ResilientSTTAdapter(
            primary_provider=primary,
            fallback_provider=fallback,
            fallback_enabled=fallback_enabled,
        )

    elif name in ("local", "whisper_local", "local_whisper"):
        return LocalWhisperSTTAdapter(
            model_name=cfg.stt_local_model,
            timeout_seconds=cfg.timeout_seconds,
            language=cfg.default_language,
        )

    elif name in ("whisper", "groq", "openai"):
        if not cfg.stt_api_key:
            raise STTUnavailableError(f"STT API key is not configured for provider '{name}'.")
        return WhisperSTTAdapter(
            api_key=cfg.stt_api_key,
            base_url=cfg.stt_base_url,
            timeout_seconds=cfg.timeout_seconds,
        )

    raise STTUnavailableError(f"Unsupported STT provider '{name}'.")
