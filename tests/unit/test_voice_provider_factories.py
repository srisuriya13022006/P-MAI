"""
P14 — Unit Tests for Voice Provider Factories.
Verifies provider dispatch matrix across mock, local, and cloud providers,
credential requirements, and structured exception handling.
"""
import pytest

from app.voice.config import VoiceConfig
from app.voice.exceptions import STTUnavailableError, TTSUnavailableError
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.stt.mock import MockSTTProvider
from app.voice.stt.whisper_adapter import WhisperSTTAdapter
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.mock import MockTTSProvider
from app.voice.tts.openai_adapter import OpenAITTSAdapter


def test_stt_factory_mock_dispatch():
    cfg = VoiceConfig(stt_provider="mock")
    provider = create_stt_provider(cfg)
    assert isinstance(provider, MockSTTProvider)


def test_stt_factory_local_dispatch():
    for name in ("local", "whisper_local", "local_whisper", "default", "", None):
        cfg = VoiceConfig(stt_provider=name, stt_local_model="tiny")
        provider = create_stt_provider(cfg)
        assert isinstance(provider, LocalWhisperSTTAdapter)
        assert provider.model_name == "tiny"


def test_stt_factory_cloud_dispatch_with_key():
    for name in ("whisper", "groq", "openai"):
        cfg = VoiceConfig(stt_provider=name, stt_api_key="test-key-12345")
        provider = create_stt_provider(cfg)
        assert isinstance(provider, WhisperSTTAdapter)


def test_stt_factory_cloud_dispatch_missing_key():
    for name in ("whisper", "groq", "openai"):
        cfg = VoiceConfig(stt_provider=name, stt_api_key=None)
        with pytest.raises(STTUnavailableError) as exc_info:
            create_stt_provider(cfg)
        assert "API key is not configured" in str(exc_info.value)


def test_stt_factory_unsupported_provider():
    cfg = VoiceConfig(stt_provider="unknown_stt_provider")
    with pytest.raises(STTUnavailableError) as exc_info:
        create_stt_provider(cfg)
    assert "Unsupported STT provider" in str(exc_info.value)


def test_tts_factory_mock_dispatch():
    cfg = VoiceConfig(tts_provider="mock")
    provider = create_tts_provider(cfg)
    assert isinstance(provider, MockTTSProvider)


def test_tts_factory_local_dispatch():
    for name in ("local", "pyttsx3", "local_tts", "default", "", None):
        cfg = VoiceConfig(tts_provider=name, tts_local_engine="pyttsx3")
        provider = create_tts_provider(cfg)
        assert isinstance(provider, LocalTTSAdapter)
        assert provider.engine_name == "pyttsx3"


def test_tts_factory_cloud_dispatch_with_key():
    for name in ("openai", "openai_tts"):
        cfg = VoiceConfig(tts_provider=name, tts_api_key="test-key-54321")
        provider = create_tts_provider(cfg)
        assert isinstance(provider, OpenAITTSAdapter)


def test_tts_factory_cloud_dispatch_missing_key():
    for name in ("openai", "openai_tts"):
        cfg = VoiceConfig(tts_provider=name, tts_api_key=None)
        with pytest.raises(TTSUnavailableError) as exc_info:
            create_tts_provider(cfg)
        assert "API key is not configured" in str(exc_info.value)


def test_tts_factory_unsupported_provider():
    cfg = VoiceConfig(tts_provider="unsupported_tts_provider")
    with pytest.raises(TTSUnavailableError) as exc_info:
        create_tts_provider(cfg)
    assert "Unsupported TTS provider" in str(exc_info.value)
