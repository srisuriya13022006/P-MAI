"""
P13 / P14 — Text-to-Speech Provider Factory.
Instantiates configured local, production cloud, or mock TTS adapters.
"""
from app.voice.config import VoiceConfig
from app.voice.exceptions import TTSUnavailableError
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.mock import MockTTSProvider
from app.voice.tts.openai_adapter import OpenAITTSAdapter


def create_tts_provider(config: VoiceConfig | None = None) -> TextToSpeechProvider:
    """
    Create configured TTS provider instance.
    Supports 'mock', 'local', and 'openai'.
    """
    cfg = config or VoiceConfig()
    name = (cfg.tts_provider or "").lower().strip()

    if name == "mock":
        return MockTTSProvider(audio_content_type=cfg.tts_audio_format)

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
