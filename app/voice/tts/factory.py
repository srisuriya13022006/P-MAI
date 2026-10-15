"""
P13 / P14 — Text-to-Speech Provider Factory.
Instantiates configured local, production cloud, or mock TTS adapters.
"""
from app.voice.config import VoiceConfig
from app.voice.exceptions import TTSUnavailableError
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.mock import MockTTSProvider
from app.voice.tts.openai_adapter import OpenAITTSAdapter
from app.voice.tts.resilient_adapter import ResilientTTSAdapter


def create_tts_provider(config: VoiceConfig | None = None) -> TextToSpeechProvider:
    """
    Create configured TTS provider instance.
    Supports 'elevenlabs' (primary with local fallback), 'local', 'openai', and 'mock'.
    """
    cfg = config or VoiceConfig.from_settings()
    name = (cfg.tts_provider or "").lower().strip()

    if name == "mock":
        return MockTTSProvider(audio_content_type=cfg.tts_audio_format)

    elif name in ("elevenlabs", "eleven_labs", "eleven"):
        primary = ElevenLabsTTSAdapter(
            api_key=cfg.elevenlabs_api_key,
            voice_id=cfg.elevenlabs_voice_id,
            model_id=cfg.elevenlabs_model_id,
            stability=cfg.elevenlabs_stability,
            similarity_boost=cfg.elevenlabs_similarity_boost,
            style=cfg.elevenlabs_style,
            use_speaker_boost=cfg.elevenlabs_use_speaker_boost,
            optimize_streaming_latency=cfg.elevenlabs_optimize_streaming_latency,
            timeout_seconds=cfg.timeout_seconds,
        )
        if cfg.tts_fallback_enabled:
            fallback = LocalTTSAdapter(
                engine_name=cfg.tts_local_engine,
                default_voice=cfg.default_voice,
                timeout_seconds=cfg.timeout_seconds,
            )
            return ResilientTTSAdapter(
                primary_provider=primary,
                fallback_provider=fallback,
                fallback_enabled=True,
            )
        return primary

    elif name in ("local", "pyttsx3", "local_tts", "default", ""):
        return LocalTTSAdapter(
            engine_name=cfg.tts_local_engine,
            default_voice=cfg.default_voice,
            timeout_seconds=cfg.timeout_seconds,
        )

    elif name in ("openai", "openai_tts"):
        if not cfg.tts_api_key:
            raise TTSUnavailableError(f"TTS API key is not configured for provider '{name}'.")
        return OpenAITTSAdapter(
            api_key=cfg.tts_api_key,
            base_url=cfg.tts_base_url,
            default_voice=cfg.default_voice,
            timeout_seconds=cfg.timeout_seconds,
        )

    raise TTSUnavailableError(f"Unsupported TTS provider '{name}'.")

