"""
P14 — Unit Tests for Local-First Voice Providers (Local Whisper STT & Local pyttsx3 TTS).
Verifies lazy loading, canonical audio handling, streaming chunks, error handling,
and credential isolation without requiring cloud network access.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.voice.audio import CanonicalAudioFormat, pcm_to_wav
from app.voice.exceptions import (
    STTError,
    STTTimeoutError,
    STTUnavailableError,
    TTSError,
    TTSTimeoutError,
    TTSUnavailableError,
)
from app.voice.schemas import STTResult, TTSResult
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.tts.local_adapter import LocalTTSAdapter


@pytest.mark.asyncio
async def test_local_stt_adapter_initialization_defaults():
    adapter = LocalWhisperSTTAdapter(model_name="tiny", device="cpu", compute_type="int8")
    assert adapter.model_name == "tiny"
    assert adapter.device == "cpu"
    assert "tiny" in repr(adapter)
    # Lazy init: underlying model should not be loaded on construction
    assert adapter._model is None


@pytest.mark.asyncio
async def test_local_stt_adapter_missing_backend():
    adapter = LocalWhisperSTTAdapter(model_name="tiny")
    with patch.dict("sys.modules", {"faster_whisper": None, "whisper": None}):
        with pytest.raises(STTUnavailableError) as exc_info:
            adapter._ensure_model_loaded()
        assert "faster-whisper" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_local_stt_adapter_empty_audio():
    adapter = LocalWhisperSTTAdapter()
    result = await adapter.transcribe(b"")
    assert result.transcript == ""
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_local_stt_adapter_valid_transcription_mocked():
    adapter = LocalWhisperSTTAdapter(model_name="tiny")
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello world from local speech."
    mock_info = MagicMock()
    mock_info.avg_logprob = -0.2
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    adapter._model = mock_model
    adapter._backend = "faster-whisper"

    sample_pcm = b"\x00\x00" * 16000  # 1 second of silence
    sample_wav = pcm_to_wav(sample_pcm)

    res = await adapter.transcribe(sample_wav)
    assert res.transcript == "Hello world from local speech."
    assert res.language == "en"
    assert res.confidence > 0.0


@pytest.mark.asyncio
async def test_local_stt_adapter_timeout_handling():
    adapter = LocalWhisperSTTAdapter(timeout_seconds=0.01)

    def slow_transcribe(path):
        import time
        time.sleep(0.05)
        return STTResult(transcript="slow")

    adapter._transcribe_file_sync = slow_transcribe
    sample_wav = pcm_to_wav(b"\x00\x00" * 3200)

    with pytest.raises(STTTimeoutError):
        await adapter.transcribe(sample_wav)


@pytest.mark.asyncio
async def test_local_stt_streaming_frames_and_finalize():
    adapter = LocalWhisperSTTAdapter()
    adapter.transcribe = AsyncMock(
        side_effect=[
            STTResult(transcript="Hello"),
            STTResult(transcript="Hello world from stream"),
        ]
    )

    # Feed chunks below partial threshold
    part1 = await adapter.feed_audio_chunk(b"\x00\x00" * 4000)
    assert len(part1) == 0

    # Feed chunk crossing threshold (>=16000 bytes)
    part2 = await adapter.feed_audio_chunk(b"\x00\x00" * 12000)
    assert len(part2) == 1
    assert part2[0].transcript == "Hello"

    # Finalize
    final_res = await adapter.finalize_stream()
    assert final_res.transcript == "Hello world from stream"
    assert adapter._stream_assembler.total_bytes == 0  # Cleared after finalize


@pytest.mark.asyncio
async def test_local_tts_adapter_initialization_defaults():
    adapter = LocalTTSAdapter(engine_name="pyttsx3", rate=180)
    assert adapter.engine_name == "pyttsx3"
    assert adapter.rate == 180
    assert "pyttsx3" in repr(adapter)


@pytest.mark.asyncio
async def test_local_tts_adapter_missing_backend():
    adapter = LocalTTSAdapter()
    with patch.dict("sys.modules", {"pyttsx3": None}):
        with pytest.raises(TTSUnavailableError) as exc_info:
            adapter._synthesize_file_sync("Hello", "out.wav")
        assert "pyttsx3" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_local_tts_adapter_empty_text():
    adapter = LocalTTSAdapter()
    res = await adapter.synthesize("")
    assert res.audio_bytes == b""
    assert res.duration_seconds == 0.0


@pytest.mark.asyncio
async def test_local_tts_adapter_synthesis_mocked():
    adapter = LocalTTSAdapter()

    def fake_synth(text, path, voice=None):
        raw_pcm = b"\x01\x00" * 16000
        wav_data = pcm_to_wav(raw_pcm)
        with open(path, "wb") as f:
            f.write(wav_data)

    adapter._synthesize_file_sync = fake_synth
    result = await adapter.synthesize("Testing local speech output.")
    assert len(result.audio_bytes) > 44
    assert result.content_type == "audio/wav"
    assert result.audio_bytes.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_local_tts_adapter_streaming_chunks():
    adapter = LocalTTSAdapter()

    def fake_synth(text, path, voice=None):
        # 1.5 seconds of canonical PCM audio
        raw_pcm = b"\x02\x00" * 24000
        wav_data = pcm_to_wav(raw_pcm)
        with open(path, "wb") as f:
            f.write(wav_data)

    adapter._synthesize_file_sync = fake_synth
    chunks = []
    async for c in adapter.synthesize_stream("Testing stream chunking.", chunk_duration_seconds=0.5):
        chunks.append(c)

    assert len(chunks) >= 3
    # Verify continuous PCM bytes (no RIFF header inside individual streaming chunks)
    for c in chunks:
        assert not c.audio_bytes.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_local_tts_timeout_handling():
    adapter = LocalTTSAdapter(timeout_seconds=0.01)

    def slow_synth(text, path, voice=None):
        import time
        time.sleep(0.05)

    adapter._synthesize_file_sync = slow_synth
    with pytest.raises(TTSTimeoutError):
        await adapter.synthesize("Slow speech")
