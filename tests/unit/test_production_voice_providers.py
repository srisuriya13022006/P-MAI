"""
Unit tests for P13 Production Voice Provider Adapters (Whisper and OpenAI TTS).
Tests configuration, missing credentials, mocked network calls, timeouts,
reconnect limits, interruption, and credential leakage prevention.
"""
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from app.voice.config import VoiceConfig
from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTTimeoutError,
    STTUnavailableError,
    TTSTimeoutError,
    TTSUnavailableError,
)
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.whisper_adapter import WhisperSTTAdapter
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.openai_adapter import OpenAITTSAdapter


# -------------------------------------------------------------------------
# 1. Provider Configuration & Missing Credentials
# -------------------------------------------------------------------------
def test_provider_configuration_defaults():
    cfg = VoiceConfig()
    assert cfg.stt_provider in ("local", "assemblyai")
    assert cfg.tts_provider == "local"
    assert cfg.sample_rate == 16000
    assert cfg.audio_format == "audio/wav"


def test_missing_credentials_stt():
    cfg = VoiceConfig(stt_provider="whisper", stt_api_key=None)
    with pytest.raises(STTUnavailableError) as exc_info:
        create_stt_provider(cfg)
    assert "STT API key is not configured" in str(exc_info.value)


def test_missing_credentials_tts():
    cfg = VoiceConfig(tts_provider="openai", tts_api_key=None)
    with pytest.raises(TTSUnavailableError) as exc_info:
        create_tts_provider(cfg)
    assert "TTS API key is not configured" in str(exc_info.value)


# -------------------------------------------------------------------------
# 2. Whisper STT Adapter - Success
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_whisper_stt_adapter_success():
    adapter = WhisperSTTAdapter(api_key="secret-test-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "What time is it in Tokyo?"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await adapter.transcribe(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 30)

        assert res.transcript == "What time is it in Tokyo?"
        assert res.provider == "whisper"
        assert res.confidence == 1.0

        # Verify Authorization header was passed
        mock_post.assert_called_once()
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-test-key"


# -------------------------------------------------------------------------
# 3. Whisper STT Adapter - Timeout
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_whisper_stt_adapter_timeout():
    adapter = WhisperSTTAdapter(api_key="secret-test-key", max_reconnect_attempts=1)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        with pytest.raises(STTTimeoutError):
            await adapter.transcribe(b"VALID_BYTES_DATA")


# -------------------------------------------------------------------------
# 4. Whisper STT Adapter - Reconnect Limit
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_whisper_stt_adapter_reconnect_limit():
    adapter = WhisperSTTAdapter(api_key="secret-test-key", max_reconnect_attempts=2)

    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 503

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_err_resp
        with pytest.raises(STTUnavailableError) as exc_info:
            await adapter.transcribe(b"VALID_BYTES_DATA")

        # Must have attempted 3 times (1 initial + 2 retries)
        assert mock_post.call_count == 3
        assert "503" in str(exc_info.value)


# -------------------------------------------------------------------------
# 5. OpenAI TTS Adapter - Success
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tts_adapter_success():
    adapter = OpenAITTSAdapter(api_key="secret-tts-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"SYNTHESIZED_WAV_AUDIO_BYTES"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await adapter.synthesize(text="Hello world")

        assert res.audio_bytes == b"SYNTHESIZED_WAV_AUDIO_BYTES"
        assert res.provider == "openai_tts"
        assert res.content_type == "audio/wav"

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-tts-key"


# -------------------------------------------------------------------------
# 6. OpenAI TTS Adapter - Timeout
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tts_adapter_timeout():
    adapter = OpenAITTSAdapter(api_key="secret-tts-key", max_reconnect_attempts=1)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("TTS timed out")
        with pytest.raises(TTSTimeoutError):
            await adapter.synthesize(text="Hello world")


# -------------------------------------------------------------------------
# 7. OpenAI TTS Adapter - Interruption / Stream Early Break
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tts_adapter_streaming_chunks():
    adapter = OpenAITTSAdapter(api_key="secret-tts-key")

    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = 200

    async def fake_chunks(chunk_size=4096):
        yield b"CHUNK_1_"
        yield b"CHUNK_2_"
        yield b"CHUNK_3_"

    mock_stream_resp.aiter_bytes = fake_chunks

    # Mock the client.stream async context manager
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream.return_value = mock_stream_ctx

    with patch("httpx.AsyncClient", return_value=mock_client):
        chunks = []
        async for chunk in adapter.synthesize_stream("Long response to interrupt"):
            chunks.append(chunk.audio_bytes)
            if len(chunks) == 2:
                # User barge-in: break early
                break

        assert len(chunks) == 2
        assert chunks[0] == b"CHUNK_1_"
        assert chunks[1] == b"CHUNK_2_"


# -------------------------------------------------------------------------
# 8. Credential Leakage Prevention
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_credential_leakage_prevention():
    secret_key = "sk-SUPER-SECRET-CREDENTIAL-12345"
    stt_adapter = WhisperSTTAdapter(api_key=secret_key, max_reconnect_attempts=0)

    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 401

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_err_resp
        with pytest.raises(STTUnavailableError) as exc_info:
            await stt_adapter.transcribe(b"SOME_AUDIO")

        # Secret key must NEVER appear in the error message
        assert secret_key not in str(exc_info.value)
        assert secret_key not in repr(exc_info.value)
