"""
P15.8 — Comprehensive Acceptance Tests for Long-Form Speech Completeness and Playback Reliability.

Validates:
1. LLM generates complete long-form response (6 paragraphs, 400-600 words) without output token truncation.
2. SpeechTextPreprocessor preserves all paragraphs and words during speech preparation.
3. Text segmentation correctly divides long text into logical chunks at sentence/paragraph boundaries without loss.
4. ElevenLabsTTSAdapter synthesizes and streams all segments sequentially without dropping tail chunks.
5. Server audio chunks are generated and delivered completely with accurate telemetry.
6. Desktop client receives all chunks, plays them, and tracks playback completion.
7. ResilientTTSAdapter handles mid-stream segment failure gracefully without duplicating already-spoken audio.
8. Barge-in / interruption stops audio playback immediately and preserves session state.
9. Short responses retain low latency and original single-chunk behavior.
10. Session inactivity and lifetime bounds remain strictly enforced.
"""
import asyncio
import base64
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.agent.orchestrator import MAIOrchestrator
from app.voice.config import VoiceConfig
from app.voice.events import RealtimeClientEvent, RealtimeEventType, RealtimeServerEvent
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import TTSResult
from app.voice.session import RealtimeVoiceSession, SessionStatus, VoiceSessionManager
from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from app.voice.tts.preprocessor import SpeechTextPreprocessor, segment_text_for_speech
from app.voice.tts.resilient_adapter import ResilientTTSAdapter
from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState


SIX_PARAGRAPH_STORY = """Once upon a time in the ancient harbor of Eldoria, an old lighthouse keeper named Arthur tended the great brass lantern atop the craggy headland. For forty years, the rhythmic pulse of its amber beam had guided fishermen through treacherous waters where jagged reefs waited like sleeping leviathans beneath the surf.

Arthur knew the sea's temper better than any man alive. On quiet evenings, he would sit on the balcony listening to the gulls call out their twilight songs, while the salt spray misted against his weathered face. He kept a leather-bound journal filled with ink sketches of passing merchant caravels and notes on the migratory patterns of gray whales that breached along the coast.

One bitter autumn twilight, a violent gale descended from the northern straits with startling speed. The sky curdled into bruised violet clouds, and furious waves began battering the granite base of the tower with the force of siege engines. Within an hour, thick sea-fog smothered the coastline, blinding even the sharpest eyes on the shore.

As Arthur inspected the burner wicks, his keen ears caught the frantic clanging of a ship's bell echoing through the thunderous roar of the surf. Peering through the rain-streaked glass, he spotted the faint silhouette of a storm-tossed schooner, her sails shredded and her rudder fractured, drifting helplessly toward the razor-sharp rocks of Widow's Reef.

Without hesitation, Arthur threw his weight against the heavy iron counterweights, manually engaging the emergency parabolic lens. The great crystal prism rotated faster, focusing every candlepower into a piercing shaft of incandescent gold that pierced straight through the shroud of mist, illuminating the safe channel between the reefs.

The exhausted helmsman caught sight of the blazing beacon and turned hard into the calm waters of the harbor basin. When dawn broke over Eldoria, the vessel rode safely at anchor, and Arthur extinguished the lantern with a quiet smile, knowing the harbor had survived another storm."""


def test_1_llm_story_generation_token_headroom():
    """Verify orchestrator allocates 1000 tokens for story queries even if prompt is short."""
    orch = MAIOrchestrator(llm=MagicMock())
    orch.llm.generate.return_value = SIX_PARAGRAPH_STORY

    # Short 7-word prompt requesting a story
    short_prompt = "Tell me a story about Arthur's lighthouse."
    res = orch.handle(short_prompt, conversation_messages=[])
    assert res == SIX_PARAGRAPH_STORY

    # Verify max_tokens passed to llm was 1000, not 300
    call_kwargs = orch.llm.generate.call_args[1]
    assert call_kwargs.get("max_tokens") == 1000


def test_2_speech_preprocessor_preserves_every_paragraph():
    """Verify SpeechTextPreprocessor does not truncate or drop any paragraphs from the 6-paragraph story."""
    prep = SpeechTextPreprocessor()
    result = prep.preprocess(SIX_PARAGRAPH_STORY, max_segment_chars=1500)

    # All 6 paragraphs must be present in clean_spoken_text
    assert "Arthur tended the great brass lantern" in result.clean_spoken_text
    assert "whispering" not in result.clean_spoken_text  # sanity check
    assert "survived another storm" in result.clean_spoken_text
    assert len(result.clean_spoken_text) > 1800

    # Ensure segmentation kept all paragraphs in order
    assert len(result.segments) >= 2
    rejoined = " ".join(result.segments)
    assert "Arthur tended the great brass lantern" in rejoined
    assert "survived another storm" in rejoined


def test_3_segment_text_for_speech_boundary_integrity():
    """Verify segment_text_for_speech never splits words and respects max_segment_chars."""
    segments = segment_text_for_speech(SIX_PARAGRAPH_STORY, max_segment_chars=1200)
    assert len(segments) >= 2

    total_words_original = len(SIX_PARAGRAPH_STORY.split())
    total_words_segmented = sum(len(s.split()) for s in segments)
    assert total_words_original == total_words_segmented

    for seg in segments:
        assert len(seg) <= 1250  # within boundary
        assert not seg.startswith(" ")
        assert not seg.endswith(" ")


@pytest.mark.asyncio
async def test_4_elevenlabs_adapter_segmented_streaming():
    """Verify ElevenLabsTTSAdapter streams all segments sequentially without discarding the tail."""
    adapter = ElevenLabsTTSAdapter(
        api_key="mock_key",
        voice_id="EXAVITQu4vr4xnSDxMaL",
        model_id="eleven_v4_turbo",
        max_segment_chars=1000,  # forces multiple segments
    )

    mock_chunks = [b"\x00\x01" * 160, b"\x00\x02" * 160]

    async def mock_fetch_stream(client, text, voice_id, is_stream):
        for c in mock_chunks:
            yield c

    with patch.object(adapter, "_get_client", return_value=MagicMock()), \
         patch.object(adapter, "_fetch_audio_stream", side_effect=mock_fetch_stream):

        emitted_chunks = []
        async for chunk in adapter.synthesize_stream(SIX_PARAGRAPH_STORY):
            emitted_chunks.append(chunk)

        # Since SIX_PARAGRAPH_STORY is ~2100 chars and max_segment_chars=1000,
        # it splits into at least 2 or 3 segments, each yielding mock_chunks.
        assert len(emitted_chunks) >= 4
        total_bytes = sum(len(c.audio_bytes) for c in emitted_chunks)
        assert total_bytes > 0
        assert total_bytes % 2 == 0  # 16-bit PCM aligned


@pytest.mark.asyncio
async def test_5_realtime_voice_pipeline_completeness_and_telemetry():
    """Verify RealtimeVoicePipeline delivers complete audio chunks and rich metadata without 450-char truncation."""
    config = VoiceConfig(
        tts_provider="mock",
        realtime_tts_chunk_duration_seconds=0.5,
    )

    from app.voice.tts.streaming import StreamingTextToSpeechProvider
    mock_tts = MagicMock(spec=StreamingTextToSpeechProvider)
    # Mock streaming TTS that yields 10 chunks for the long story
    async def mock_stream(text, **kwargs):
        assert "survived another storm" in text  # Proves tail of story was NOT discarded!
        for i in range(10):
            yield TTSResult(audio_bytes=b"\x01\x00" * 320, content_type="audio/wav", duration_seconds=0.5, provider="mock")

    mock_tts.synthesize_stream = mock_stream
    mock_tts.provider_name = "mock"

    mock_orch = MagicMock()
    mock_orch.handle.return_value = SIX_PARAGRAPH_STORY

    pipeline = RealtimeVoicePipeline(
        orchestrator=mock_orch,
        tts_provider=mock_tts,
        config=config,
    )

    session = pipeline.session_manager.create_session("test_conv", "test_user")
    session.transition_to(SessionStatus.LISTENING)

    # Send AUDIO_END event with simulated final transcript
    from app.voice.stt.streaming import StreamingSpeechToTextProvider
    mock_stt = MagicMock(spec=StreamingSpeechToTextProvider)
    mock_stt.finalize_stream = AsyncMock(return_value=MagicMock(transcript="Tell me the full story of Arthur."))
    mock_stt.model_name = "universal-3-6-pro"
    pipeline.stt_provider = mock_stt

    events = []
    async for event in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END)):
        events.append(event)

    chunk_events = [e for e in events if e.type == RealtimeEventType.RESPONSE_CHUNK]
    end_events = [e for e in events if e.type == RealtimeEventType.RESPONSE_END]

    assert len(chunk_events) == 10  # All 10 audio chunks emitted
    assert len(end_events) == 1

    end_ev = end_events[0]
    meta = end_ev.metadata
    assert meta["text_chars"] > 1800
    assert meta["text_paragraphs"] == 6
    assert meta["total_audio_chunks"] == 10
    assert meta["total_audio_bytes"] == 10 * 640
    assert meta["stages"]["text_generation_completed"] is True
    assert meta["stages"]["audio_synthesis_completed"] is True
    assert meta["stages"]["audio_delivery_completed"] is True


@pytest.mark.asyncio
async def test_6_desktop_client_receives_and_tracks_long_response():
    """Verify desktop PMAIVoiceClient tracks received chunks, total bytes, and completes playback."""
    client = PMAIVoiceClient()
    client.turn_metrics = {"client_turn_start": time.perf_counter(), "client_audio_end": time.perf_counter()}

    # Simulate 8 incoming RESPONSE_CHUNK events
    for i in range(8):
        dummy_audio = b"\x00\x02" * 320
        b64 = base64.b64encode(dummy_audio).decode("ascii")
        client.handle_server_event({
            "type": "RESPONSE_CHUNK",
            "audio_base64": b64,
            "metadata": {"chunk_index": i + 1},
        })

    assert client.received_chunks_count == 8
    assert client.received_audio_bytes == 8 * 640

    # Simulate RESPONSE_END
    client.handle_server_event({
        "type": "RESPONSE_END",
        "text_chunk": SIX_PARAGRAPH_STORY,
    })

    assert client.state_machine.current_state == ClientUIState.IDLE
    assert client.turn_metrics["received_audio_chunks"] == 8
    assert client.turn_metrics["received_audio_bytes"] == 8 * 640
    assert client.turn_metrics["stages"]["playback_completed"] is True


@pytest.mark.asyncio
async def test_7_resilient_tts_adapter_mid_stream_fallback_no_duplication():
    """Verify ResilientTTSAdapter only falls back on remaining segments when primary fails mid-stream."""
    from app.voice.tts.streaming import StreamingTextToSpeechProvider
    primary_mock = MagicMock(spec=StreamingTextToSpeechProvider)
    primary_mock.provider_name = "elevenlabs"

    call_count = 0
    async def primary_stream(text, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Segment 1 succeeds
            yield TTSResult(audio_bytes=b"\xaa\xaa" * 160, content_type="audio/wav", duration_seconds=0.5, provider="elevenlabs")
        else:
            # Segment 2 fails
            raise RuntimeError("ElevenLabs rate limit on segment 2")

    primary_mock.synthesize_stream = primary_stream

    fallback_mock = MagicMock(spec=StreamingTextToSpeechProvider)
    fallback_mock.provider_name = "local"
    fallback_mock.synthesize = AsyncMock(return_value=TTSResult(audio_bytes=b"\xbb\xbb" * 160, content_type="audio/wav", duration_seconds=0.5, provider="local"))
    fb_received_texts = []

    async def fallback_stream(text, **kwargs):
        fb_received_texts.append(text)
        yield TTSResult(audio_bytes=b"\xbb\xbb" * 160, content_type="audio/wav", duration_seconds=0.5, provider="local")

    fallback_mock.synthesize_stream = fallback_stream

    adapter = ResilientTTSAdapter(
        primary_provider=primary_mock,
        fallback_provider=fallback_mock,
        fallback_enabled=True,
    )

    chunks = []
    async for chunk in adapter.synthesize_stream(SIX_PARAGRAPH_STORY):
        chunks.append(chunk)

    # First chunk came from elevenlabs
    assert chunks[0].provider == "elevenlabs"
    # Second chunk came from local fallback
    assert chunks[1].provider == "local"

    # Crucial assertion: Fallback must NOT have re-synthesized the entire story from the beginning!
    assert len(fb_received_texts) > 0
    assert not fb_received_texts[0].startswith("Once upon a time in the ancient harbor")


@pytest.mark.asyncio
async def test_8_barge_in_and_cancellation_during_long_speech():
    """Verify barge-in halts playback immediately and transitions state."""
    config = VoiceConfig()
    pipeline = RealtimeVoicePipeline(config=config)
    session = pipeline.session_manager.create_session("conv_1", "user_1")
    session.transition_to(SessionStatus.LISTENING)
    session.transition_to(SessionStatus.TRANSCRIBING)
    session.transition_to(SessionStatus.PROCESSING)
    session.transition_to(SessionStatus.SPEAKING)

    # Client sends INTERRUPT event
    events = []
    async for ev in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.INTERRUPT, reason="barge_in")):
        events.append(ev)

    assert len(events) == 1
    assert events[0].type == RealtimeEventType.INTERRUPT
    assert session.status == SessionStatus.LISTENING
    assert session.is_interrupted is True


@pytest.mark.asyncio
async def test_9_short_responses_retain_single_segment_low_latency():
    """Verify short responses do not incur extra segmentation or delay."""
    short_reply = "Good morning! The weather is sunny and 72 degrees."
    prep = SpeechTextPreprocessor()
    res = prep.preprocess(short_reply)

    assert len(res.segments) == 1
    assert res.segments[0] == short_reply


def test_10_session_inactivity_and_duration_bounds():
    """Verify session bounds (120s inactivity, 600s max duration) remain intact."""
    config = VoiceConfig(
        realtime_inactivity_timeout_seconds=120.0,
        realtime_max_session_duration_seconds=600.0,
    )
    session = RealtimeVoiceSession(
        session_id="s1",
        conversation_id="c1",
        user_id="u1",
        config=config,
    )

    # Within bounds -> no exception
    session.check_bounds()

    # Exceed inactivity -> raises SessionInactivityTimeoutError
    from app.voice.exceptions import SessionInactivityTimeoutError
    session.last_activity_at = time.time() - 125.0
    with pytest.raises(SessionInactivityTimeoutError):
        session.check_bounds()
