"""
Integration tests for WebSocket /voice/realtime endpoint and regression verification.
Covers real-time streaming exchange, barge-in, error frames, and backward compatibility.
"""
import base64
import json
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
from app.voice.events import RealtimeEventType
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


@pytest.fixture
def sample_chunk_b64() -> str:
    audio = generate_mock_wav(duration_seconds=0.2)
    return base64.b64encode(audio).decode("ascii")


def test_realtime_websocket_full_turn(client, sample_chunk_b64):
    """Test full duplex streaming turn via WebSocket."""
    with patch("app.services.chat_service.ChatService.chat", return_value="The time in Tokyo is 2:00 AM."):
        with client.websocket_connect("/voice/realtime?conversation_id=ws-test-conv") as websocket:
            # 1. AUDIO_START
            websocket.send_json({"type": "AUDIO_START"})
            resp_start = websocket.receive_json()
            assert resp_start["type"] == RealtimeEventType.SPEECH_START

            # 2. AUDIO_CHUNK
            websocket.send_json({
                "type": "AUDIO_CHUNK",
                "audio_base64": sample_chunk_b64,
            })
            # May receive partial transcript
            chunk_resp = websocket.receive_json()
            assert chunk_resp["type"] in (
                RealtimeEventType.TRANSCRIPT_PARTIAL,
                RealtimeEventType.AUDIO_CHUNK,
            )

            # 3. AUDIO_END
            websocket.send_json({"type": "AUDIO_END"})

            # Read sequence until RESPONSE_END
            event_types = []
            while True:
                msg = websocket.receive_json()
                event_types.append(msg["type"])
                if msg["type"] == RealtimeEventType.RESPONSE_END:
                    assert msg["text_chunk"] == "The time in Tokyo is 2:00 AM."
                    break

            assert RealtimeEventType.SPEECH_END in event_types
            assert RealtimeEventType.TRANSCRIPT_FINAL in event_types
            assert RealtimeEventType.RESPONSE_START in event_types
            assert RealtimeEventType.RESPONSE_CHUNK in event_types

            # 4. CLOSE
            websocket.send_json({"type": "CLOSE"})
            close_resp = websocket.receive_json()
            assert close_resp["type"] == RealtimeEventType.CLOSE


def test_realtime_websocket_barge_in(client, sample_chunk_b64):
    """Test client-initiated interruption / barge-in over WebSocket."""
    with client.websocket_connect("/voice/realtime") as websocket:
        websocket.send_json({"type": "AUDIO_START"})
        msg1 = websocket.receive_json()
        assert msg1["type"] == RealtimeEventType.SPEECH_START

        # Send INTERRUPT event
        websocket.send_json({"type": "INTERRUPT", "reason": "user_pressed_stop"})
        msg2 = websocket.receive_json()
        assert msg2["type"] == RealtimeEventType.INTERRUPT
        assert msg2["is_interrupted"] is True

        websocket.send_json({"type": "CLOSE"})


def test_37_post_voice_chat_regression(client):
    """Requirement 37: Verify POST /voice/chat continues to work unchanged."""
    audio_bytes = generate_mock_wav(duration_seconds=1.0)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    with patch("app.services.chat_service.ChatService.chat", return_value="4,000."):
        payload = {
            "conversation_id": "regress-voice-conv",
            "audio_base64": audio_b64,
            "content_type": "audio/wav",
        }
        res = client.post("/voice/chat", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["text_response"] == "4,000."
        assert data["audio_base64"] is not None


def test_38_post_chat_regression(client):
    """Requirement 38: Verify existing POST /chat continues to work completely intact."""
    with patch("app.services.chat_service.ChatService.chat", return_value="Hello from standard text chat."):
        payload = {
            "user_id": "test-user",
            "conversation_id": "regress-text-conv",
            "message": "Hello P-MAI",
        }
        res = client.post("/chat", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["response"] == "Hello from standard text chat."
