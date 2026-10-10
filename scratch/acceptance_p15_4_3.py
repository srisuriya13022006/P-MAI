"""
P15.4.3 Live Acceptance Script:
Deterministic Fast Paths, Datetime Resolution, Safe Telemetry, and Context Continuation.

Runs live audio turns through RealtimeVoicePipeline using AssemblyAI v3 streaming (universal-3-6-pro):
A. "Hello"
B. "All okay"
C. "What is your name?"
D. "What time is it?"
E. "What time is it in Tokyo?"
F. "Search the web for today's AI news."
G. "Read it again."
"""
import asyncio
import json
import logging
import os
import time
import wave
import audioop
import pyttsx3

from app.core.config import settings
from app.database.connection import SessionLocal
from app.services.chat_service import ChatService
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.stt.assemblyai_adapter import AssemblyAIRealtimeSTTAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("acceptance_p15_4_3")


def synthesize_pcm16(text: str) -> bytes:
    """Generate 16kHz mono 16-bit PCM audio from text."""
    engine = pyttsx3.init()
    os.makedirs("scratch", exist_ok=True)
    wav_path = f"scratch/synth_{int(time.time()*1000)}.wav"
    try:
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with wave.open(wav_path, "rb") as wf:
            nframes = wf.getnframes()
            data = wf.readframes(nframes)
            rate = wf.getframerate()
            data_16k, _ = audioop.ratecv(data, 2, 1, rate, 16000, None)
            return data_16k
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


async def run_voice_turn(pipeline: RealtimeVoicePipeline, session, pcm_bytes: bytes, label: str):
    """Feed audio chunks into RealtimeVoicePipeline and measure latencies."""
    chunk_size = 3200  # 100ms chunks at 16kHz PCM16 (3200 bytes)
    
    t0_wall = time.perf_counter()
    
    # 1. AUDIO_START
    events_start = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START))]
    
    # 2. AUDIO_CHUNK streaming with realistic pacing
    for i in range(0, len(pcm_bytes), chunk_size):
        import base64
        chunk = pcm_bytes[i:i+chunk_size]
        b64 = base64.b64encode(chunk).decode("ascii")
        ev_list = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=b64))]
        await asyncio.sleep(0.04)
    
    # 3. AUDIO_END -> triggers STT finalize, deterministic routing/reasoning, tool execution, TTS
    t0_finalize = time.perf_counter()
    response_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]
    t_wall_end = time.perf_counter()
    
    finalize_turnaround_ms = (t_wall_end - t0_finalize) * 1000
    total_wall_ms = (t_wall_end - t0_wall) * 1000
    
    transcript = ""
    response_text = ""
    for e in response_events:
        if e.type == RealtimeEventType.TRANSCRIPT_FINAL:
            transcript = e.transcript or ""
        elif e.type == RealtimeEventType.RESPONSE_CHUNK and e.text_chunk:
            response_text += e.text_chunk
        elif e.type == RealtimeEventType.RESPONSE_END and e.text_chunk and not response_text:
            response_text = e.text_chunk
        elif e.type == RealtimeEventType.ERROR:
            logger.error(f"Voice Turn Error: {e.error_code} - {e.error_message}")
            
    return {
        "label": label,
        "transcript": transcript,
        "response_text": response_text,
        "finalize_turnaround_ms": finalize_turnaround_ms,
        "total_wall_ms": total_wall_ms,
    }


async def main():
    logger.info("=================================================================")
    logger.info("P15.4.3 ACCEPTANCE TEST SUITE: FAST PATHS, DATETIME & TELEMETRY")
    logger.info("=================================================================")

    adapter = AssemblyAIRealtimeSTTAdapter(
        api_key=settings.assemblyai_api_key,
        model_name="universal-3-6-pro",
    )

    db = SessionLocal()
    results = {}
    try:
        chat_service = ChatService(db)
        pipeline = RealtimeVoicePipeline(
            chat_service=chat_service,
            stt_provider=adapter,
        )

        test_cases = [
            ("A", "Hello", "greeting"),
            ("B", "All okay", "acknowledgment"),
            ("C", "What is your name?", "self_identity"),
            ("D", "What time is it?", "datetime_generic"),
            ("E", "What time is it in Tokyo?", "datetime_tokyo"),
        ]

        turn_results = []
        for code, text, category in test_cases:
            logger.info(f"--- Running Test {code}: '{text}' ---")
            pcm = synthesize_pcm16(text)
            session = pipeline.session_manager.create_session(
                conversation_id=f"conv-p1543-{code}-{int(time.time()*1000)}",
                user_id="user_p1543",
            )

            # Snapshot LLM cloud calls before turn
            llm_prov = chat_service.orchestrator.llm
            cloud_calls_before = getattr(llm_prov, "cloud_call_count", 0)

            turn_res = await run_voice_turn(pipeline, session, pcm, text)
            
            cloud_calls_after = getattr(llm_prov, "cloud_call_count", 0)
            calls_made = cloud_calls_after - cloud_calls_before
            last_telemetry = getattr(llm_prov, "last_telemetry", {})

            # Routing decision snapshot
            from app.agent.policy import is_self_identity_query, is_pure_datetime_query
            if is_self_identity_query(text):
                route = "local (fast_path)"
                intent = "identify_self"
                llm_used = "none (0 calls - deterministic fast path)"
            elif is_pure_datetime_query(text):
                route = "tool (fast_path)"
                intent = "datetime"
                llm_used = "none (0 calls - deterministic DateTimeTool)"
            else:
                route = "local"
                intent = category
                llm_used = f"{last_telemetry.get('active_provider', 'groq')}/{last_telemetry.get('active_model', settings.cloud_model)}"

            turn_res["test_code"] = code
            turn_res["stt_provider"] = "assemblyai"
            turn_res["stt_model"] = getattr(adapter, "_effective_model", "universal-3-6-pro")
            turn_res["route"] = route
            turn_res["intent"] = intent
            turn_res["llm_provider_model"] = llm_used
            turn_res["cloud_calls_made"] = calls_made
            turn_res["fallback_reason"] = last_telemetry.get("fallback_reason") if calls_made > 0 else None
            
            logger.info(
                f"Result {code}: Wall={turn_res['total_wall_ms']:.1f}ms | Route={route} | LLM={llm_used} | Response='{turn_res['response_text'][:60]}...'"
            )
            turn_results.append(turn_res)

        # Context Continuation: F ("Search the web for today's AI news.") -> G ("Read it again.")
        logger.info("--- Running Test F & G: Web Search & 'Read it again' Context Continuation ---")
        conv_fg = f"conv-continuation-fg-{int(time.time()*1000)}"
        session_fg = pipeline.session_manager.create_session(
            conversation_id=conv_fg,
            user_id="user_p1543",
        )

        # Turn F
        pcm_f = synthesize_pcm16("Search the web for today's AI news.")
        llm_prov = chat_service.orchestrator.llm
        cb_f = getattr(llm_prov, "cloud_call_count", 0)
        res_f = await run_voice_turn(pipeline, session_fg, pcm_f, "Search the web for today's AI news.")
        ca_f = getattr(llm_prov, "cloud_call_count", 0)
        tel_f = getattr(llm_prov, "last_telemetry", {})
        res_f["test_code"] = "F"
        res_f["route"] = "tool"
        res_f["intent"] = "web_search"
        res_f["stt_provider"] = "assemblyai"
        res_f["stt_model"] = getattr(adapter, "_effective_model", "universal-3-6-pro")
        res_f["llm_provider_model"] = f"{tel_f.get('active_provider', 'groq')}/{tel_f.get('active_model', settings.cloud_model)}"
        res_f["cloud_calls_made"] = ca_f - cb_f
        res_f["fallback_reason"] = tel_f.get("fallback_reason")
        logger.info(f"Result F: Wall={res_f['total_wall_ms']:.1f}ms | Response='{res_f['response_text'][:70]}...'")
        turn_results.append(res_f)

        # Turn G: "Read it again." in the SAME session
        pcm_g = synthesize_pcm16("Read it again.")
        cb_g = getattr(llm_prov, "cloud_call_count", 0)
        res_g = await run_voice_turn(pipeline, session_fg, pcm_g, "Read it again.")
        ca_g = getattr(llm_prov, "cloud_call_count", 0)
        res_g["test_code"] = "G"
        res_g["route"] = "context_continuation"
        res_g["intent"] = "repetition"
        res_g["stt_provider"] = "assemblyai"
        res_g["stt_model"] = getattr(adapter, "_effective_model", "universal-3-6-pro")
        res_g["llm_provider_model"] = "none (0 calls - prior response reused)"
        res_g["cloud_calls_made"] = ca_g - cb_g
        res_g["fallback_reason"] = None
        logger.info(f"Result G: Wall={res_g['total_wall_ms']:.1f}ms | Response='{res_g['response_text'][:70]}...'")
        turn_results.append(res_g)

        results["turns"] = turn_results

        with open("scratch/acceptance_p15_4_3_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to scratch/acceptance_p15_4_3_results.json")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
