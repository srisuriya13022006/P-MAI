"""
P13 / P14 — Speech-to-Text Provider Factory.
Instantiates configured local, production cloud, or mock STT adapters.
"""
from app.voice.config import VoiceConfig
from app.voice.exceptions import STTUnavailableError
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.stt.mock import MockSTTProvider
from app.voice.stt.whisper_adapter import WhisperSTTAdapter


def create_stt_provider(config: VoiceConfig | None = None) -> SpeechToTextProvider:
    """
    Create configured STT provider instance.
    Supports 'mock', 'local', 'whisper', 'groq', and 'openai'.
    """
    cfg = config or VoiceConfig()
    name = (cfg.stt_provider or "").lower().strip()

    if name == "mock":
        return MockSTTProvider()

    elif name in ("local", "whisper_local", "local_whisper", "default", ""):
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
