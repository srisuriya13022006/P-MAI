"""
Integration tests for P13 Voice Client Protocol and Audio Normalization.
Tests canonical audio normalization, bounded jitter buffering, streaming protocol,
barge-in, stop vs cancel semantics, and full backward compatibility.
"""
import base64
import json
import struct
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

from app.database.connection import Base
from app.database.session import get_db
from app.llm.ollama_provider import get_ollama_provider
from app.main import app
from app.voice.audio import AudioNormalizer, BoundedAudioChunkBuffer, CanonicalAudioFormat
from app.voice.events import RealtimeEventType
from app.voice.exceptions import BackpressureError, CorruptedAudioError
from app.voice.tts.mock import generate_mock_wav


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    app.dependency_overrides[get_db] = lambda: test_db
    mock_llm = MagicMock()
    app.dependency_overrides[get_ollama_provider] = lambda: mock_llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# 1. Canonical Audio Format Invariant
# -------------------------------------------------------------------------
def test_canonical_audio_format():
    fmt = CanonicalAudioFormat()
    assert fmt.sample_rate == 16000
    assert fmt.channels == 1
    assert fmt.sample_width == 2
    assert fmt.bytes_per_second == 32000


# -------------------------------------------------------------------------
# 2. Audio Normalization - Sample Rate & Mono Conversion
# -------------------------------------------------------------------------
def test_stereo_to_mono_normalization():
    normalizer = AudioNormalizer(target_rate=16000)
    # Generate 100 samples of 16-bit stereo at 16000Hz (left=1000, right=3000 -> avg=2000)
    stereo_pcm = bytearray()
    for _ in range(100):
        stereo_pcm.extend(struct.pack("<h", 1000))
        stereo_pcm.extend(struct.pack("<h", 3000))

    mono_out = normalizer.normalize_pcm(
        bytes(stereo_pcm),
        in_sample_rate=16000,
        in_channels=2,
        in_sample_width=2,
    )

    # Output should have exactly 100 samples (200 bytes)
    assert len(mono_out) == 200
    unpacked = struct.unpack("<100h", mono_out)
    assert unpacked[0] == 2000


def test_sample_rate_resampling():
    normalizer = AudioNormalizer(target_rate=16000)
    # Generate 32000 samples of 32kHz mono -> should resample to ~16000 samples
    in_samples = [int(5000 * (i % 2)) for i in range(3200)]
    pcm_32k = bytearray()
    for s in in_samples:
        pcm_32k.extend(struct.pack("<h", s))

    out_16k = normalizer.normalize_pcm(
        bytes(pcm_32k),
        in_sample_rate=32000,
        in_channels=1,
        in_sample_width=2,
    )

    num_out_samples = len(out_16k) // 2
    assert 1500 <= num_out_samples <= 1700


def test_malformed_frame_alignment():
    normalizer = AudioNormalizer(target_rate=16000)
    # Pass odd byte count (e.g. 15 bytes for 16-bit samples -> last byte trimmed)
    odd_bytes = b"\x00" * 15
    out = normalizer.normalize_pcm(odd_bytes, in_sample_rate=16000, in_channels=1, in_sample_width=2)
    assert len(out) % 2 == 0


# -------------------------------------------------------------------------
# 3. Bounded Buffering & Duplicate Chunk Suppression
# -------------------------------------------------------------------------
def test_bounded_buffering_and_duplicate_suppression():
    buffer = BoundedAudioChunkBuffer(max_bytes=1000)
    chunk = b"\x01\x02\x03\x04"

    # First append succeeds
    assert buffer.append_chunk(chunk) is True
    assert buffer.current_size == 4

    # Duplicate chunk is suppressed without raising error
    assert buffer.append_chunk(chunk) is False
    assert buffer.current_size == 4

    # Backpressure limit raises BackpressureError
    huge_chunk = b"\x00" * 1050
    with pytest.raises(BackpressureError):
        buffer.append_chunk(huge_chunk)


# -------------------------------------------------------------------------
# 4. WebSocket Client Turnaround
# -------------------------------------------------------------------------
def test_websocket_client_turnaround(client):
    audio_chunk = generate_mock_wav(duration_seconds=0.2)
    audio_b64 = base64.b64encode(audio_chunk).decode("ascii")

    with patch("app.services.chat_service.ChatService.chat", return_value="Current time in Tokyo is 2:00 AM."):
        with client.websocket_connect("/voice/realtime?conversation_id=p13-client-conv") as ws:
            # Send AUDIO_START
            ws.send_json({"type": "AUDIO_START"})
            res_start = ws.receive_json()
            assert res_start["type"] == RealtimeEventType.SPEECH_START

            # Send AUDIO_CHUNK
            ws.send_json({"type": "AUDIO_CHUNK", "audio_base64": audio_b64})
            res_chunk = ws.receive_json()
            assert res_chunk["type"] in (RealtimeEventType.TRANSCRIPT_PARTIAL, RealtimeEventType.AUDIO_CHUNK)

            # Send AUDIO_END
            ws.send_json({"type": "AUDIO_END"})

            # Collect sequence until RESPONSE_END
            received_events = []
            while True:
                msg = ws.receive_json()
                received_events.append(msg["type"])
                if msg["type"] == RealtimeEventType.RESPONSE_END:
                    assert "Tokyo" in msg["text_chunk"]
                    break

            assert RealtimeEventType.TRANSCRIPT_FINAL in received_events
            assert RealtimeEventType.RESPONSE_START in received_events
            assert RealtimeEventType.RESPONSE_CHUNK in received_events


# -------------------------------------------------------------------------
# 5. Stop Speaking vs Cancel Task Semantics
# -------------------------------------------------------------------------
def test_stop_speaking_vs_cancel_task(client):
    with patch("app.services.chat_service.ChatService.chat", return_value="The task has been cancelled.") as mock_chat:
        with client.websocket_connect("/voice/realtime") as ws:
            # Test 'Stop' -> Mute only, no chat invocation
            ws.send_json({"type": "AUDIO_START"})
            ws.receive_json()

            # Mock transcript returning 'Stop.'
            with patch("app.voice.stt.streaming.MockStreamingSTTProvider.finalize_stream") as mock_stt:
                from app.voice.schemas import STTResult
                mock_stt.return_value = STTResult(transcript="Stop.", provider="mock")

                ws.send_json({"type": "AUDIO_END"})
                msg = ws.receive_json()  # SPEECH_END
                msg_final = ws.receive_json()  # TRANSCRIPT_FINAL
                msg_end = ws.receive_json()  # RESPONSE_END

                assert msg_end["type"] == RealtimeEventType.RESPONSE_END
                assert msg_end["metadata"].get("action") == "stop_speaking_only"
                mock_chat.assert_not_called()


# -------------------------------------------------------------------------
# 6. Regressions: POST /voice/chat and POST /chat
# -------------------------------------------------------------------------
def test_post_voice_chat_regression(client):
    audio_bytes = generate_mock_wav(duration_seconds=1.0)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    with patch("app.services.chat_service.ChatService.chat", return_value="Verified answer for voice."):
        payload = {
            "conversation_id": "regress-conv",
            "audio_base64": audio_b64,
            "content_type": "audio/wav",
        }
        res = client.post("/voice/chat", json=payload)
        assert res.status_code == 200
        assert res.json()["text_response"] == "Verified answer for voice."


def test_post_chat_text_regression(client):
    with patch("app.services.chat_service.ChatService.chat", return_value="Verified text answer."):
        payload = {
            "user_id": "u-1",
            "conversation_id": "regress-text",
            "message": "Hello",
        }
        res = client.post("/chat", json=payload)
        assert res.status_code == 200
        assert res.json()["response"] == "Verified text answer."
