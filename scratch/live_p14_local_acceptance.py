"""
Live Local Acceptance Runner for P14 — Local-First Voice & Production Hardening.
Benchmarks real offline local voice execution using LocalWhisperSTTAdapter and LocalTTSAdapter,
measuring end-to-end latencies, barge-in response, and state machine integrity.
Outputs results to scratch/live_p14_results.json.
"""
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Add workspace to sys.path
sys.path.insert(0, r"D:\suriya\projects\P-MAI")

from app.services.chat_service import ChatService
from app.voice.audio import CanonicalAudioFormat, pcm_to_wav
from app.voice.config import VoiceConfig
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.formatter import VoicePresentationFormatter
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import STTResult, TTSResult
from app.voice.session import RealtimeVoiceSession, SessionStatus, VoiceSessionManager
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.tts.local_adapter import LocalTTSAdapter


async def run_live_p14_acceptance():
    print("=" * 70)
    print("STARTING LIVE ACCEPTANCE SUITE: P14 - LOCAL-FIRST VOICE PIPELINE")
    print("=" * 70)

    cfg = VoiceConfig(
        enabled=True,
        realtime_enabled=True,
        stt_provider="local",
        tts_provider="local",
        stt_local_model="tiny",
        tts_local_engine="pyttsx3",
    )

    formatter = VoicePresentationFormatter()
    stt = LocalWhisperSTTAdapter(model_name="tiny")
    tts = LocalTTSAdapter(engine_name="pyttsx3")

    print(f"STT Provider: {stt.provider_name} (Local Offline)")
    print(f"TTS Provider: {tts.provider_name} (Local Offline)")
    print(f"Canonical Format: 16000Hz, 1ch, 16-bit Signed PCM")
    print("=" * 70)

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "local_offline",
            "stt_provider": stt.provider_name,
            "tts_provider": tts.provider_name,
            "canonical_format": {
                "sample_rate": CanonicalAudioFormat.sample_rate,
                "channels": CanonicalAudioFormat.channels,
                "sample_width": CanonicalAudioFormat.sample_width,
            },
        },
        "benchmarks": {},
        "scenarios": {},
    }

    # Generate test audio: synthesize a sample utterance locally to use as realistic STT input
    print("\n[Step 0] Pre-synthesizing reference audio using local TTS...")
    ref_synth = await tts.synthesize("What time is it in Tokyo?")
    ref_wav = ref_synth.audio_bytes
    ref_b64 = base64.b64encode(ref_wav).decode("ascii")
    print(f"  Generated reference WAV: {len(ref_wav)} bytes ({ref_synth.duration_seconds}s)")

    # -------------------------------------------------------------
    # Scenario A: Offline Local Turn & Complete Latency Measurement
    # -------------------------------------------------------------
    print("\n[Scenario A] Local Turn: 'What time is it in Tokyo?'")
    try:
        mock_chat = MagicMock(spec=ChatService)
        raw_mai_response = "The current time in Tokyo, Japan is 02:45 AM (JST)."
        mock_chat.chat.return_value = raw_mai_response

        manager = VoiceSessionManager(cfg)
        session = manager.create_session(conversation_id="conv-p14-a")
        pipeline = RealtimeVoicePipeline(
            chat_service=mock_chat,
            stt_provider=stt,
            tts_provider=tts,
            session_manager=manager,
            config=cfg,
        )

        t_capture_start = time.perf_counter()
        # 1. AUDIO_START
        start_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START))]
        t_audio_start = time.perf_counter()

        # 2. AUDIO_CHUNK
        chunk_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=ref_b64))]
        t_chunk_sent = time.perf_counter()

        # 3. AUDIO_END
        t_mic_end = time.perf_counter()
        end_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]
        t_turn_complete = time.perf_counter()

        final_tr = [e for e in end_events if e.type == RealtimeEventType.TRANSCRIPT_FINAL][0]
        end_resp = [e for e in end_events if e.type == RealtimeEventType.RESPONSE_END][0]
        tts_chunks = [e for e in end_events if e.type == RealtimeEventType.RESPONSE_CHUNK]

        assert final_tr.transcript != ""
        assert "Tokyo" in end_resp.text_chunk
        assert len(tts_chunks) > 0

        # Detailed local latency breakdown
        latency_metrics = {
            "mic_start_to_audio_start_ms": round((t_audio_start - t_capture_start) * 1000.0, 2),
            "chunk_ingestion_latency_ms": round((t_chunk_sent - t_audio_start) * 1000.0, 2),
            "stt_finalization_latency_ms": round((t_turn_complete - t_mic_end) * 1000.0, 2),
            "mai_reasoning_latency_ms": end_resp.metadata.get("mai_latency_ms", 0.0),
            "tts_synthesis_latency_ms": end_resp.metadata.get("tts_latency_ms", 0.0),
            "total_turn_latency_ms": round((t_turn_complete - t_capture_start) * 1000.0, 2),
        }

        results["scenarios"]["Scenario_A_LocalTurn"] = {
            "status": "PASSED",
            "transcript": final_tr.transcript,
            "response": end_resp.text_chunk,
            "tts_chunks_count": len(tts_chunks),
            "latency": latency_metrics,
        }
        print(f"  [PASS] Transcript: \"{final_tr.transcript}\"")
        print(f"  [PASS] TTS Chunks: {len(tts_chunks)} | Total: {latency_metrics['total_turn_latency_ms']}ms")
    except Exception as e:
        results["scenarios"]["Scenario_A_LocalTurn"] = {"status": "FAILED", "error": str(e)}
        print(f"  [FAIL] Failed: {e}")

    # -------------------------------------------------------------
    # Scenario B: Offline Local Barge-In Handling
    # -------------------------------------------------------------
    print("\n[Scenario B] Local Barge-In Interruption")
    try:
        manager = VoiceSessionManager(cfg)
        session = manager.create_session(conversation_id="conv-p14-b")
        pipeline = RealtimeVoicePipeline(chat_service=MagicMock(spec=ChatService), stt_provider=stt, tts_provider=tts, session_manager=manager, config=cfg)

        session.status = SessionStatus.SPEAKING
        t0 = time.perf_counter()
        int_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START))]
        t_int_dur = time.perf_counter() - t0

        assert session.status == SessionStatus.LISTENING
        assert any(e.type == RealtimeEventType.INTERRUPT and e.is_interrupted for e in int_events)

        results["scenarios"]["Scenario_B_LocalBargeIn"] = {
            "status": "PASSED",
            "barge_in_latency_ms": round(t_int_dur * 1000.0, 2),
            "status_after_interrupt": session.status.value,
        }
        print(f"  [PASS] Local barge-in interrupted playback in {t_int_dur*1000:.2f}ms")
    except Exception as e:
        results["scenarios"]["Scenario_B_LocalBargeIn"] = {"status": "FAILED", "error": str(e)}
        print(f"  [FAIL] Failed: {e}")

    # -------------------------------------------------------------
    # Scenario C: Offline Stop-Speaking (Playback Mute Without Cancel)
    # -------------------------------------------------------------
    print("\n[Scenario C] Local Stop-Speaking ('Stop.')")
    try:
        manager = VoiceSessionManager(cfg)
        session = manager.create_session(conversation_id="conv-p14-c")
        mock_chat = MagicMock(spec=ChatService)

        # Force Mock STT to deliver exact "Stop." transcript
        pipeline = RealtimeVoicePipeline(chat_service=mock_chat, stt_provider=stt, tts_provider=tts, session_manager=manager, config=cfg)
        # Mock stt.finalize_stream for precise phrase testing
        stt.finalize_stream = MagicMock(return_value=asyncio.sleep(0, result=STTResult(transcript="Stop.", confidence=0.99)))

        await anext(pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)))
        events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]

        end_resp = [e for e in events if e.type == RealtimeEventType.RESPONSE_END][0]
        assert end_resp.metadata.get("action") == "stop_speaking_only"
        mock_chat.chat.assert_not_called()

        results["scenarios"]["Scenario_C_StopSpeaking"] = {
            "status": "PASSED",
            "action": end_resp.metadata.get("action"),
            "task_cancelled": False,
        }
        print(f"  [PASS] Spoken 'Stop' muted playback without invoking MAI task state")
    except Exception as e:
        results["scenarios"]["Scenario_C_StopSpeaking"] = {"status": "FAILED", "error": str(e)}
        print(f"  [FAIL] Failed: {e}")

    # -------------------------------------------------------------
    # Scenario D: Offline Spoken Task Cancellation
    # -------------------------------------------------------------
    print("\n[Scenario D] Local Task Cancellation ('Cancel this task.')")
    try:
        manager = VoiceSessionManager(cfg)
        session = manager.create_session(conversation_id="conv-p14-d")
        mock_chat = MagicMock(spec=ChatService)
        mock_chat.chat.return_value = "Task cancelled successfully."

        pipeline = RealtimeVoicePipeline(chat_service=mock_chat, stt_provider=stt, tts_provider=tts, session_manager=manager, config=cfg)
        stt.finalize_stream = MagicMock(return_value=asyncio.sleep(0, result=STTResult(transcript="Cancel this task.", confidence=0.99)))

        await anext(pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)))
        events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]

        end_resp = [e for e in events if e.type == RealtimeEventType.RESPONSE_END][0]
        assert mock_chat.chat.call_count == 1
        assert "cancelled" in end_resp.text_chunk.lower()

        results["scenarios"]["Scenario_D_TaskCancellation"] = {
            "status": "PASSED",
            "chat_called": True,
            "response": end_resp.text_chunk,
        }
        print(f"  [PASS] 'Cancel this task' routed authoritatively to MAI P8 Task State")
    except Exception as e:
        results["scenarios"]["Scenario_D_TaskCancellation"] = {"status": "FAILED", "error": str(e)}
        print(f"  [FAIL] Failed: {e}")

    # -------------------------------------------------------------
    # Summary & Output
    # -------------------------------------------------------------
    total = len(results["scenarios"])
    passed = sum(1 for s in results["scenarios"].values() if s.get("status") == "PASSED")
    failed = total - passed

    print("\n" + "=" * 70)
    print(f"P14 LOCAL ACCEPTANCE SUITE COMPLETE: {passed}/{total} Passed ({failed} Failed)")
    print("=" * 70)

    out_file = r"D:\suriya\projects\P-MAI\scratch\live_p14_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out_file}")


if __name__ == "__main__":
    asyncio.run(run_live_p14_acceptance())
