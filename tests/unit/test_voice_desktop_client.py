"""
P15 — Dedicated Voice Client Test Suite.
Tests all 32 mandatory requirements specified in P15:
1. initial connection
2. connection failure
3. reconnect
4. reconnect exhaustion
5. session close
6. microphone permission denial
7. microphone unavailable
8. AUDIO_START
9. AUDIO_CHUNK
10. AUDIO_END
11. partial transcript display
12. final transcript display
13. response start
14. response audio chunk handling
15. response completion
16. playback interruption
17. stop-speaking
18. cancel-task
19. barge-in
20. P8 multi-turn continuation
21. P9 clarification
22. memory confirmation
23. P11 recovery result
24. malformed server event
25. oversized client chunk
26. bounded audio buffering
27. session expiry
28. duplicate event handling
29. cleanup after disconnect
30. no server credentials exposed
31. no internal IDs displayed
32. conversation history limit
"""
import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from client.voice.audio_device import AudioDevice
from client.voice.client import PMAIVoiceClient, scrub_internal_identifiers
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState, VoiceClientStateMachine


@pytest.fixture
def mock_audio_device():
    dev = AudioDevice()
    # Mock audio streams so hardware is not required in tests
    dev.open_input_stream = MagicMock()
    return dev


@pytest.fixture
def voice_client(mock_audio_device):
    config = VoiceClientConfig(
        backend_ws_url="ws://localhost:8000/voice/realtime",
        max_reconnect_attempts=3,
        reconnect_delay_seconds=0.01,
        max_history_turns=20,
    )
    client = PMAIVoiceClient(config=config, audio_device=mock_audio_device)
    return client


# ---------------------------------------------------------------------------
# 1. Initial Connection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_01_initial_connection(voice_client):
    mock_ws = AsyncMock()
    with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
        await voice_client.connect()
        assert voice_client.is_connected is True
        assert voice_client.state_machine.current_state == ClientUIState.IDLE
        assert "connection_latency_ms" in voice_client.session_metrics


# ---------------------------------------------------------------------------
# 2. Connection Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_02_connection_failure(voice_client):
    with patch("websockets.connect", side_effect=OSError("Connection refused")):
        with pytest.raises(ConnectionError):
            await voice_client.connect()
        assert voice_client.is_connected is False
        assert voice_client.state_machine.current_state == ClientUIState.ERROR


# ---------------------------------------------------------------------------
# 3. Reconnect
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_03_reconnect(voice_client):
    mock_ws = AsyncMock()
    attempts = 0

    async def mock_connect_impl(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("fail 1")
        return mock_ws

    with patch("websockets.connect", side_effect=mock_connect_impl):
        success = await voice_client.reconnect()
        assert success is True
        assert voice_client.is_connected is True
        assert voice_client.state_machine.current_state == ClientUIState.IDLE


# ---------------------------------------------------------------------------
# 4. Reconnect Exhaustion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_04_reconnect_exhaustion(voice_client):
    with patch("websockets.connect", side_effect=OSError("Host down")):
        success = await voice_client.reconnect()
        assert success is False
        assert voice_client.is_connected is False
        assert voice_client.state_machine.current_state == ClientUIState.DISCONNECTED
        assert "Connection lost" in voice_client.state_machine.error_message


# ---------------------------------------------------------------------------
# 5. Session Close
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_05_session_close(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    await voice_client.close()
    assert voice_client.is_connected is False
    assert voice_client.state_machine.current_state == ClientUIState.CLOSED
    mock_ws.send.assert_awaited()
    mock_ws.close.assert_awaited()


# ---------------------------------------------------------------------------
# 6. Microphone Permission Denial
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_06_microphone_permission_denial(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True
    voice_client.device.permission_granted = False

    with pytest.raises(PermissionError) as exc_info:
        await voice_client.start_turn()
    assert "permission denied" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 7. Microphone Unavailable
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_07_microphone_unavailable(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True
    voice_client.device.device_available = False

    with pytest.raises(RuntimeError) as exc_info:
        await voice_client.start_turn()
    assert "not available" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 8. AUDIO_START
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_08_audio_start(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    await voice_client.start_turn()
    assert voice_client.state_machine.current_state == ClientUIState.LISTENING
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "AUDIO_START"


# ---------------------------------------------------------------------------
# 9. AUDIO_CHUNK
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_09_audio_chunk(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    chunk = voice_client.device.generate_synthetic_chunk()
    await voice_client.send_audio_chunk(chunk)
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "AUDIO_CHUNK"
    assert "audio_base64" in sent
    assert voice_client.buffered_audio_bytes == len(chunk)


# ---------------------------------------------------------------------------
# 10. AUDIO_END
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_audio_end(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    await voice_client.finish_turn()
    assert voice_client.state_machine.current_state == ClientUIState.TRANSCRIBING
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["type"] == "AUDIO_END"


# ---------------------------------------------------------------------------
# 11. Partial Transcript Display
# ---------------------------------------------------------------------------
def test_11_partial_transcript_display(voice_client):
    cb = MagicMock()
    voice_client.on_partial_transcript = cb

    voice_client.handle_server_event({
        "type": "TRANSCRIPT_PARTIAL",
        "transcript": "What is the time",
    })
    assert voice_client.current_partial_transcript == "What is the time"
    cb.assert_called_with("What is the time")
    # Verify not added to conversation history
    assert len(voice_client.history) == 0


# ---------------------------------------------------------------------------
# 12. Final Transcript Display
# ---------------------------------------------------------------------------
def test_12_final_transcript_display(voice_client):
    cb = MagicMock()
    voice_client.on_final_transcript = cb

    voice_client.handle_server_event({
        "type": "TRANSCRIPT_FINAL",
        "transcript": "What is the time in Tokyo?",
    })
    assert voice_client.current_final_transcript == "What is the time in Tokyo?"
    assert voice_client.state_machine.current_state == ClientUIState.PROCESSING
    assert len(voice_client.history) == 1
    assert voice_client.history[0]["speaker"] == "User"
    assert voice_client.history[0]["text"] == "What is the time in Tokyo?"
    cb.assert_called_with("What is the time in Tokyo?")


# ---------------------------------------------------------------------------
# 13. Response Start
# ---------------------------------------------------------------------------
def test_13_response_start(voice_client):
    voice_client.handle_server_event({"type": "RESPONSE_START"})
    assert voice_client.state_machine.current_state == ClientUIState.PROCESSING


# ---------------------------------------------------------------------------
# 14. Response Audio Chunk Handling
# ---------------------------------------------------------------------------
def test_14_response_audio_chunk_handling(voice_client):
    chunk = voice_client.device.generate_synthetic_chunk()
    b64 = base64.b64encode(chunk).decode("ascii")

    voice_client.handle_server_event({
        "type": "RESPONSE_CHUNK",
        "audio_base64": b64,
    })
    assert voice_client.state_machine.current_state == ClientUIState.SPEAKING


# ---------------------------------------------------------------------------
# 15. Response Completion
# ---------------------------------------------------------------------------
def test_15_response_completion(voice_client):
    cb = MagicMock()
    voice_client.on_turn_completed = cb

    voice_client.handle_server_event({
        "type": "RESPONSE_END",
        "text_chunk": "The time in Tokyo is 2:00 AM.",
        "metadata": {"mai_latency_ms": 120.0},
    })
    assert voice_client.current_response_text == "The time in Tokyo is 2:00 AM."
    assert voice_client.state_machine.current_state == ClientUIState.IDLE
    assert len(voice_client.history) == 1
    assert voice_client.history[0]["speaker"] == "MAI"
    assert voice_client.history[0]["text"] == "The time in Tokyo is 2:00 AM."


# ---------------------------------------------------------------------------
# 16. Playback Interruption
# ---------------------------------------------------------------------------
def test_16_playback_interruption(voice_client):
    chunk = voice_client.device.generate_synthetic_chunk()
    voice_client.device.stop_playback()
    # Playback should be skipped/rejected when interrupted
    played = voice_client.device.play_audio_chunk(chunk)
    assert played is False


# ---------------------------------------------------------------------------
# 17. Stop Speaking (Preserves Task)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_stop_speaking(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    with patch.object(voice_client.device, "stop_playback") as mock_stop_pb:
        await voice_client.stop_speaking()
        mock_stop_pb.assert_called_once()
        assert voice_client.state_machine.current_state == ClientUIState.IDLE
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "INTERRUPT"
        assert sent["reason"] == "stop_speaking_button"


# ---------------------------------------------------------------------------
# 18. Cancel Task
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_18_cancel_task(voice_client):
    mock_ws = AsyncMock()
    voice_client.ws = mock_ws
    voice_client.is_connected = True

    await voice_client.cancel_task()
    # Verify AUDIO_START and AUDIO_END were sent
    calls = [json.loads(c[0][0])["type"] for c in mock_ws.send.call_args_list]
    assert "AUDIO_START" in calls
    assert "AUDIO_CHUNK" in calls
    assert "AUDIO_END" in calls


# ---------------------------------------------------------------------------
# 19. Barge-in (Immediate Interruption on Speech Start)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_19_barge_in(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True
    voice_client.state_machine.transition_to(ClientUIState.SPEAKING)

    with patch.object(voice_client.device, "stop_playback") as mock_stop_pb:
        await voice_client.start_turn()
        mock_stop_pb.assert_called_once()
        assert "interruption_latency_ms" in voice_client.turn_metrics
        assert voice_client.state_machine.current_state == ClientUIState.LISTENING


# ---------------------------------------------------------------------------
# 20. P8 Multi-turn Continuation
# ---------------------------------------------------------------------------
def test_20_p8_multi_turn_continuation(voice_client):
    # Turn 1
    voice_client.handle_server_event({"type": "TRANSCRIPT_FINAL", "transcript": "Find the latest Python release."})
    voice_client.handle_server_event({"type": "RESPONSE_END", "text_chunk": "Python 3.14.8 is the latest."})

    # Turn 2
    voice_client.handle_server_event({"type": "TRANSCRIPT_FINAL", "transcript": "Compare it with Python thirteen."})
    voice_client.handle_server_event({"type": "RESPONSE_END", "text_chunk": "Compared to 3.13, 3.14 adds..."})

    assert len(voice_client.history) == 4
    assert voice_client.history[0]["speaker"] == "User"
    assert voice_client.history[1]["speaker"] == "MAI"
    assert voice_client.history[2]["speaker"] == "User"
    assert voice_client.history[3]["speaker"] == "MAI"


# ---------------------------------------------------------------------------
# 21. P9 Clarification Display
# ---------------------------------------------------------------------------
def test_21_p9_clarification_display(voice_client):
    voice_client.handle_server_event({"type": "TRANSCRIPT_FINAL", "transcript": "Read that page."})
    voice_client.handle_server_event({
        "type": "RESPONSE_END",
        "text_chunk": "Which page would you like me to read? Please provide the URL or title.",
    })
    assert "Which page would you like me to read?" in voice_client.history[1]["text"]


# ---------------------------------------------------------------------------
# 22. Memory Confirmation Display
# ---------------------------------------------------------------------------
def test_22_memory_confirmation_display(voice_client):
    voice_client.handle_server_event({"type": "TRANSCRIPT_FINAL", "transcript": "Remember that I am learning FastAPI."})
    voice_client.handle_server_event({
        "type": "RESPONSE_END",
        "text_chunk": "I need your confirmation before I remember that you are learning FastAPI. Would you like me to save this?",
    })
    assert "confirmation" in voice_client.history[1]["text"]


# ---------------------------------------------------------------------------
# 23. P11 Recovery Result Display
# ---------------------------------------------------------------------------
def test_23_p11_recovery_result(voice_client):
    voice_client.handle_server_event({"type": "TRANSCRIPT_FINAL", "transcript": "Fetch the docs."})
    voice_client.handle_server_event({
        "type": "RESPONSE_END",
        "text_chunk": "The initial search was unavailable, but I recovered using the alternative cache.",
    })
    assert "recovered" in voice_client.history[1]["text"]


# ---------------------------------------------------------------------------
# 24. Malformed Server Event
# ---------------------------------------------------------------------------
def test_24_malformed_server_event(voice_client):
    voice_client.handle_server_event("not a dictionary")
    assert voice_client.state_machine.current_state == ClientUIState.ERROR
    assert "malformed" in voice_client.state_machine.error_message.lower()


# ---------------------------------------------------------------------------
# 25. Oversized Client Chunk
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_25_oversized_client_chunk(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True

    huge_chunk = b"\x00" * (128 * 1024)  # 128 KB > max 64 KB
    with pytest.raises(ValueError) as exc:
        await voice_client.send_audio_chunk(huge_chunk)
    assert "exceeds maximum" in str(exc.value)


# ---------------------------------------------------------------------------
# 26. Bounded Audio Buffering
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_26_bounded_audio_buffering(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True
    voice_client.buffered_audio_bytes = voice_client.config.max_audio_buffer_bytes

    small_chunk = b"\x00" * 1024
    with pytest.raises(ValueError) as exc:
        await voice_client.send_audio_chunk(small_chunk)
    assert "buffer exceeded maximum" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 27. Session Expiry
# ---------------------------------------------------------------------------
def test_27_session_expiry(voice_client):
    voice_client.handle_server_event({
        "type": "ERROR",
        "error_code": "SESSION_EXPIRED",
        "error_message": "Session has timed out.",
    })
    assert voice_client.state_machine.current_state == ClientUIState.ERROR
    assert "SESSION_EXPIRED" in voice_client.state_machine.error_message


# ---------------------------------------------------------------------------
# 28. Duplicate Event Handling
# ---------------------------------------------------------------------------
def test_28_duplicate_event_handling(voice_client):
    # Multiple identical events shouldn't cause errors
    voice_client.handle_server_event({"type": "SPEECH_START"})
    voice_client.handle_server_event({"type": "SPEECH_START"})
    assert voice_client.state_machine.current_state == ClientUIState.LISTENING


# ---------------------------------------------------------------------------
# 29. Cleanup After Disconnect
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_29_cleanup_after_disconnect(voice_client):
    voice_client.ws = AsyncMock()
    voice_client.is_connected = True
    with patch.object(voice_client.device, "close") as mock_dev_close:
        await voice_client.close()
        mock_dev_close.assert_called_once()
        assert voice_client.is_connected is False


# ---------------------------------------------------------------------------
# 30. No Server Credentials Exposed
# ---------------------------------------------------------------------------
def test_30_no_server_credentials_exposed():
    cfg = VoiceClientConfig()
    cfg_dict = cfg.model_dump()
    forbidden_keys = ["api_key", "secret", "token", "password", "key"]
    for key in cfg_dict:
        for f in forbidden_keys:
            assert f not in key.lower()


# ---------------------------------------------------------------------------
# 31. No Internal IDs Displayed
# ---------------------------------------------------------------------------
def test_31_no_internal_ids_displayed():
    raw_text = "Task task_123 failed at step_456 with trace trace_abc."
    scrubbed = scrub_internal_identifiers(raw_text)
    assert "task_123" not in scrubbed
    assert "step_456" not in scrubbed
    assert "trace_abc" not in scrubbed


# ---------------------------------------------------------------------------
# 32. Conversation History Limit
# ---------------------------------------------------------------------------
def test_32_conversation_history_limit(voice_client):
    for i in range(30):
        voice_client._append_history("User", f"Message {i}")
    assert len(voice_client.history) == voice_client.config.max_history_turns
    assert voice_client.history[-1]["text"] == "Message 29"
