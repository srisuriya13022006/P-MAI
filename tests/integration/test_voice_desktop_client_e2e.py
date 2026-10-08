"""
P15 — End-to-End Client & Realtime Backend Integration Tests.
Validates complete client/server interaction over WebSocket and HTTP:
Mock Audio -> Client WebSocket -> /voice/realtime -> MAI Engine -> Streaming TTS -> Client Playback.
"""
import base64
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

from app.database.connection import Base
from app.database.session import get_db
from app.llm.ollama_provider import get_ollama_provider
from app.main import app
from app.voice.events import RealtimeEventType
from client.voice.audio_device import AudioDevice
from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState


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
def test_app_client(test_db):
    app.dependency_overrides[get_db] = lambda: test_db
    mock_llm = MagicMock()
    app.dependency_overrides[get_ollama_provider] = lambda: mock_llm
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Web UI Endpoint
# ---------------------------------------------------------------------------
def test_get_voice_ui_endpoint(test_app_client):
    res = test_app_client.get("/voice/ui")
    assert res.status_code == 200
    assert "P-MAI Voice Assistant" in res.text
    assert "<canvas id=\"visualizer\"" in res.text


# ---------------------------------------------------------------------------
# 2. Complete End-to-End Client Turnaround
# ---------------------------------------------------------------------------
def test_e2e_voice_client_turnaround(test_app_client):
    device = AudioDevice()
    client = PMAIVoiceClient(
        config=VoiceClientConfig(conversation_id="e2e-conv-1"),
        audio_device=device,
    )

    with patch("app.services.chat_service.ChatService.chat", return_value="The time in Tokyo is 2:00 AM."):
        with test_app_client.websocket_connect("/voice/realtime?conversation_id=e2e-conv-1") as ws:
            # 1. Client starts speech
            ws.send_json({"type": "AUDIO_START"})
            evt = ws.receive_json()
            assert evt["type"] == RealtimeEventType.SPEECH_START
            client.handle_server_event(evt)
            assert client.state_machine.current_state == ClientUIState.LISTENING

            # 2. Client sends audio chunk
            chunk = device.generate_synthetic_chunk()
            b64_chunk = base64.b64encode(chunk).decode("ascii")
            ws.send_json({"type": "AUDIO_CHUNK", "audio_base64": b64_chunk})
            evt = ws.receive_json()
            client.handle_server_event(evt)

            # 3. Client finishes speech
            ws.send_json({"type": "AUDIO_END"})

            # Collect stream until completion
            received_types = []
            while True:
                msg = ws.receive_json()
                received_types.append(msg["type"])
                client.handle_server_event(msg)
                if msg["type"] == RealtimeEventType.RESPONSE_END:
                    break

            assert RealtimeEventType.SPEECH_END in received_types
            assert RealtimeEventType.TRANSCRIPT_FINAL in received_types
            assert RealtimeEventType.RESPONSE_CHUNK in received_types
            assert RealtimeEventType.RESPONSE_END in received_types

            # Verify client state returned to IDLE and conversation history has both turns
            assert client.state_machine.current_state == ClientUIState.IDLE
            assert len(client.history) >= 2
            assert client.history[-1]["speaker"] == "MAI"
            assert "Tokyo" in client.history[-1]["text"]


# ---------------------------------------------------------------------------
# 3. End-to-End Stop Speaking (Preserves Task State)
# ---------------------------------------------------------------------------
def test_e2e_stop_speaking_preserves_task(test_app_client):
    device = AudioDevice()
    client = PMAIVoiceClient(
        config=VoiceClientConfig(conversation_id="e2e-conv-stop"),
        audio_device=device,
    )

    with patch("app.services.chat_service.ChatService.chat") as mock_chat:
        with test_app_client.websocket_connect("/voice/realtime?conversation_id=e2e-conv-stop") as ws:
            # Send stop command via INTERRUPT
            ws.send_json({"type": "INTERRUPT", "reason": "stop_speaking_button"})
            evt = ws.receive_json()
            assert evt["type"] == RealtimeEventType.INTERRUPT
            assert evt["is_interrupted"] is True
            client.handle_server_event(evt)

            # Assert backend chat service was NOT called to cancel task
            mock_chat.assert_not_called()
            assert client.state_machine.current_state == ClientUIState.IDLE


# ---------------------------------------------------------------------------
# 4. End-to-End Barge-In
# ---------------------------------------------------------------------------
def test_e2e_barge_in_handling(test_app_client):
    device = AudioDevice()
    client = PMAIVoiceClient(
        config=VoiceClientConfig(conversation_id="e2e-barge"),
        audio_device=device,
    )

    with test_app_client.websocket_connect("/voice/realtime?conversation_id=e2e-barge") as ws:
        # Start turn 1
        ws.send_json({"type": "AUDIO_START"})
        ws.receive_json()  # SPEECH_START

        # Send chunk while speaking (simulate barge-in)
        chunk = device.generate_synthetic_chunk()
        b64 = base64.b64encode(chunk).decode("ascii")
        ws.send_json({"type": "AUDIO_CHUNK", "audio_base64": b64})
        chunk_evt = ws.receive_json()
        client.handle_server_event(chunk_evt)

        # Send interrupt
        ws.send_json({"type": "INTERRUPT", "reason": "barge_in_test"})
        interrupt_evt = ws.receive_json()
        assert interrupt_evt["type"] == RealtimeEventType.INTERRUPT
        client.handle_server_event(interrupt_evt)

        assert client.state_machine.current_state == ClientUIState.IDLE
