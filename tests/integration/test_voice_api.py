"""
Integration tests for Voice API endpoint: POST /voice/chat.
Verifies HTTP contract, status codes, payload handling, and privacy invariants.
"""
import base64
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
from app.voice.config import VoiceConfig
from app.voice.exceptions import STTTimeoutError, STTUnavailableError
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
def sample_audio_b64() -> str:
    audio_bytes = generate_mock_wav(duration_seconds=1.0)
    return base64.b64encode(audio_bytes).decode("ascii")


def test_post_voice_chat_success(client, sample_audio_b64):
    with patch("app.services.chat_service.ChatService.chat", return_value="The time in Tokyo is 2:00 AM."):
        payload = {
            "conversation_id": "test-v-conv",
            "user_id": "test-user",
            "audio_base64": sample_audio_b64,
            "content_type": "audio/wav",
        }
        response = client.post("/voice/chat", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["conversation_id"] == "test-v-conv"
        assert "transcript" in data
        assert data["text_response"] == "The time in Tokyo is 2:00 AM."
        assert data["audio_base64"] is not None
        assert data["status"] == "success"
        assert "timing" in data
        assert data["timing"]["stt_latency_ms"] >= 0
        assert data["timing"]["mai_latency_ms"] >= 0
        assert data["timing"]["tts_latency_ms"] >= 0

        # Privacy invariant: no internal engine traces leaked
        assert "task_id" not in data
        assert "trace_id" not in data
        assert "planner_metadata" not in data
        assert "recovery_attempts" not in data


def test_post_voice_chat_unsupported_format(client, sample_audio_b64):
    payload = {
        "conversation_id": "test-v-conv",
        "audio_base64": sample_audio_b64,
        "content_type": "audio/unsupported-xyz",
    }
    response = client.post("/voice/chat", json=payload)
    assert response.status_code == 415
    assert response.json()["detail"]["error"] == "UNSUPPORTED_AUDIO_FORMAT"


def test_post_voice_chat_corrupted_audio(client):
    payload = {
        "conversation_id": "test-v-conv",
        "audio_base64": base64.b64encode(b"CORRUPTED_NOT_WAV").decode("ascii"),
        "content_type": "audio/wav",
    }
    response = client.post("/voice/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "CORRUPTED_AUDIO"


def test_post_voice_chat_empty_audio(client):
    payload = {
        "conversation_id": "test-v-conv",
        "audio_base64": "",
        "content_type": "audio/wav",
    }
    response = client.post("/voice/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"]["error"] in ("EMPTY_AUDIO", "CORRUPTED_AUDIO")


def test_post_voice_chat_disabled(client, sample_audio_b64):
    with patch("app.voice.config.VoiceConfig.from_settings") as mock_cfg:
        mock_cfg.return_value = VoiceConfig(enabled=False)
        payload = {
            "conversation_id": "test-v-conv",
            "audio_base64": sample_audio_b64,
            "content_type": "audio/wav",
        }
        response = client.post("/voice/chat", json=payload)
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "VOICE_DISABLED"


def test_post_voice_chat_stt_unavailable(client, sample_audio_b64):
    with patch("app.voice.stt.mock.MockSTTProvider.transcribe", side_effect=STTUnavailableError("STT Down")):
        payload = {
            "conversation_id": "test-v-conv",
            "audio_base64": sample_audio_b64,
            "content_type": "audio/wav",
        }
        response = client.post("/voice/chat", json=payload)
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "STT_UNAVAILABLE"


def test_post_voice_chat_stt_timeout(client, sample_audio_b64):
    with patch("app.voice.stt.mock.MockSTTProvider.transcribe", side_effect=STTTimeoutError("STT Timeout")):
        payload = {
            "conversation_id": "test-v-conv",
            "audio_base64": sample_audio_b64,
            "content_type": "audio/wav",
        }
        response = client.post("/voice/chat", json=payload)
        assert response.status_code == 504
        assert response.json()["detail"]["error"] == "STT_TIMEOUT"
