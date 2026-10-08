"""
P14 — Integration Tests for Completely Offline Voice Pipeline.
Verifies complete local-first voice operation with LocalWhisperSTTAdapter and LocalTTSAdapter,
session state transitions, barge-in, stop-speaking, and cancellation without cloud dependencies.
"""
import base64
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.chat_service import ChatService
from app.voice.audio import CanonicalAudioFormat, pcm_to_wav
from app.voice.config import VoiceConfig
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import STTResult, TTSResult
from app.voice.session import RealtimeVoiceSession, SessionStatus, VoiceSessionManager
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.tts.local_adapter import LocalTTSAdapter


@pytest.fixture
def offline_voice_config():
    return VoiceConfig(
        enabled=True,
        realtime_enabled=True,
        stt_provider="local",
        tts_provider="local",
        stt_local_model="tiny",
        tts_local_engine="pyttsx3",
    )


@pytest.mark.asyncio
async def test_offline_complete_voice_turn(offline_voice_config):
    # Mock underlying local adapters for fast deterministic integration test
    mock_stt = MagicMock(spec=LocalWhisperSTTAdapter)
    mock_stt.feed_audio_chunk = AsyncMock(return_value=[])
    mock_stt.finalize_stream = AsyncMock(
        return_value=STTResult(transcript="What time is it in Tokyo?", confidence=0.95)
    )

    mock_tts = MagicMock(spec=LocalTTSAdapter)

    async def fake_tts_stream(text, **kwargs):
        yield TTSResult(audio_bytes=b"\x00\x00" * 4000, content_type="audio/wav")
        yield TTSResult(audio_bytes=b"\x00\x00" * 4000, content_type="audio/wav")

    mock_tts.synthesize_stream = fake_tts_stream

    mock_chat = MagicMock(spec=ChatService)
    mock_chat.chat.return_value = "The current time in Tokyo is 02:45 AM."

    session_manager = VoiceSessionManager(offline_voice_config)
    session = session_manager.create_session(conversation_id="conv-offline-1")

    pipeline = RealtimeVoicePipeline(
        chat_service=mock_chat,
        stt_provider=mock_stt,
        tts_provider=mock_tts,
        session_manager=session_manager,
        config=offline_voice_config,
    )

    # 1. AUDIO_START -> session enters LISTENING
    start_events = [
        e async for e in pipeline.handle_client_event(
            session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)
        )
    ]
    assert session.status == SessionStatus.LISTENING
    assert any(e.type == RealtimeEventType.SPEECH_START for e in start_events)

    # 2. AUDIO_CHUNK
    sample_b64 = base64.b64encode(b"\x00\x00" * 8000).decode("ascii")
    chunk_events = [
        e async for e in pipeline.handle_client_event(
            session,
            RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=sample_b64),
        )
    ]

    # 3. AUDIO_END -> TRANSCRIBING -> PROCESSING -> SPEAKING -> RESPONSE_END
    end_events = [
        e async for e in pipeline.handle_client_event(
            session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END)
        )
    ]

    # Verify event sequencing
    event_types = [e.type for e in end_events]
    assert RealtimeEventType.SPEECH_END in event_types
    assert RealtimeEventType.TRANSCRIPT_FINAL in event_types
    assert RealtimeEventType.RESPONSE_START in event_types
    assert RealtimeEventType.RESPONSE_CHUNK in event_types
    assert RealtimeEventType.RESPONSE_END in event_types

    end_resp = [e for e in end_events if e.type == RealtimeEventType.RESPONSE_END][0]
    assert "Tokyo" in end_resp.text_chunk
    assert mock_chat.chat.call_count == 1
    # Verify session returned to LISTENING (ready for next user turn)
    assert session.status in (SessionStatus.IDLE, SessionStatus.LISTENING)


@pytest.mark.asyncio
async def test_offline_barge_in_handling(offline_voice_config):
    mock_stt = MagicMock(spec=LocalWhisperSTTAdapter)
    mock_stt.finalize_stream = AsyncMock(
        return_value=STTResult(transcript="Interrupted speech", confidence=0.9)
    )
    mock_tts = MagicMock(spec=LocalTTSAdapter)

    session_manager = VoiceSessionManager(offline_voice_config)
    session = session_manager.create_session(conversation_id="conv-offline-2")
    session.status = SessionStatus.SPEAKING

    pipeline = RealtimeVoicePipeline(
        chat_service=MagicMock(spec=ChatService),
        stt_provider=mock_stt,
        tts_provider=mock_tts,
        session_manager=session_manager,
        config=offline_voice_config,
    )

    # User speaks during speaking state (barge-in)
    int_events = [
        e async for e in pipeline.handle_client_event(
            session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)
        )
    ]

    assert any(e.type == RealtimeEventType.INTERRUPT and e.is_interrupted for e in int_events)
    assert session.status == SessionStatus.LISTENING


@pytest.mark.asyncio
async def test_offline_stop_speaking_no_task_cancellation(offline_voice_config):
    mock_stt = MagicMock(spec=LocalWhisperSTTAdapter)
    mock_stt.finalize_stream = AsyncMock(
        return_value=STTResult(transcript="Stop.", confidence=0.99)
    )
    mock_tts = MagicMock(spec=LocalTTSAdapter)
    mock_chat = MagicMock(spec=ChatService)

    session_manager = VoiceSessionManager(offline_voice_config)
    session = session_manager.create_session(conversation_id="conv-offline-3")

    pipeline = RealtimeVoicePipeline(
        chat_service=mock_chat,
        stt_provider=mock_stt,
        tts_provider=mock_tts,
        session_manager=session_manager,
        config=offline_voice_config,
    )

    await anext(
        pipeline.handle_client_event(
            session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)
        )
    )
    events = [
        e async for e in pipeline.handle_client_event(
            session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END)
        )
    ]

    end_resp = [e for e in events if e.type == RealtimeEventType.RESPONSE_END][0]
    assert end_resp.metadata.get("action") == "stop_speaking_only"
    mock_chat.chat.assert_not_called()  # Muted at transport layer, P8 task preserved
