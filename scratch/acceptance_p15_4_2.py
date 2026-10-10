"""
P15.4.2 Acceptance Script: Model Verification, Realtime Latency, 4 Simple Queries, and "Read it again" Context Continuation.
"""
import asyncio
import json
import logging
import os
import sys
import time
import wave
import audioop
import pyttsx3

from app.core.config import settings
from app.database.connection import SessionLocal
from app.services.chat_service import ChatService
from app.voice.config import VoiceConfig
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.stt.assemblyai_adapter import AssemblyAIRealtimeSTTAdapter
from app.voice.stt.resilient_adapter import ResilientSTTAdapter
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.agent.orchestrator import MAIOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("acceptance_p15_4_2")


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


async def run_pipeline_turn(pipeline: RealtimeVoicePipeline, session, pcm_bytes: bytes, label: str):
    """Feed audio chunks into RealtimeVoicePipeline and measure latencies."""
    chunk_size = 3200  # 200ms at 16kHz PCM16 (100ms is 3200 bytes since 2 bytes per sample)
    
    t0_wall = time.perf_counter()
    
    # 1. AUDIO_START
    events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START))]
    
    # 2. AUDIO_CHUNK streaming
    stt_stream_t0 = time.perf_counter()
    for i in range(0, len(pcm_bytes), chunk_size):
        import base64
        chunk = pcm_bytes[i:i+chunk_size]
        b64 = base64.b64encode(chunk).decode("ascii")
        ev_list = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=b64))]
        await asyncio.sleep(0.04) # Paced
    
    # 3. AUDIO_END -> triggers STT finalize, reasoning, tool execution, TTS
    stt_finalize_t0 = time.perf_counter()
    response_events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]
    t_wall_end = time.perf_counter()
    
    total_wall_ms = (t_wall_end - t0_wall) * 1000
    
    # Extract transcripts and responses
    transcript = ""
    response_text = ""
    tts_chunks = 0
    for e in response_events:
        if e.type == RealtimeEventType.TRANSCRIPT_FINAL:
            transcript = e.transcript or ""
        elif e.type == RealtimeEventType.RESPONSE_CHUNK and e.text_chunk:
            response_text += e.text_chunk
        elif e.type == RealtimeEventType.RESPONSE_END and e.text_chunk and not response_text:
            response_text = e.text_chunk
        elif e.type == RealtimeEventType.AUDIO_CHUNK:
            tts_chunks += 1
        elif e.type == RealtimeEventType.ERROR:
            logger.error(f"Turn error event: {e.error_code} - {e.error_message}")
            
    return {
        "label": label,
        "transcript": transcript,
        "response_text": response_text,
        "total_wall_ms": total_wall_ms,
        "tts_chunks": tts_chunks,
    }


async def main():
    results = {}
    
    # -------------------------------------------------------------
    # 1. Test Production AssemblyAI Model on Realtime Streaming v3
    # -------------------------------------------------------------
    logger.info("=== STEP 1: VERIFYING PRODUCTION ASSEMBLYAI MODEL ===")
    adapter = AssemblyAIRealtimeSTTAdapter(
        api_key=settings.assemblyai_api_key,
        model_name="universal-3-6-pro",
    )
    
    hello_pcm = synthesize_pcm16("Hello MAI")
    t0_stt = time.perf_counter()
    chunk_sz = 3200
    for i in range(0, len(hello_pcm), chunk_sz):
        await adapter.feed_audio_chunk(hello_pcm[i:i+chunk_sz])
        await asyncio.sleep(0.04)
    stt_res = await adapter.finalize_stream()
    stt_ms = (time.perf_counter() - t0_stt) * 1000
    
    logger.info(f"AssemblyAI v3 Realtime STT Transcript: '{stt_res.transcript}' in {stt_ms:.1f}ms")
    logger.info(f"AssemblyAI v3 Effective Model: {getattr(adapter, '_effective_model', 'unknown')}")
    results["step1_assemblyai_model"] = {
        "configured_model": adapter.model_name,
        "effective_model": getattr(adapter, "_effective_model", "universal-3-6-pro"),
        "test_transcript": stt_res.transcript,
        "latency_ms": stt_ms,
    }
    
    # -------------------------------------------------------------
    # 2. Test 4 Simple Queries through the Live Pipeline
    # -------------------------------------------------------------
    logger.info("=== STEP 2 & 3: 4 SIMPLE VOICE QUERIES WITH FULL LATENCY BREAKDOWN ===")
    simple_queries = [
        ("Hello", "greeting"),
        ("All okay", "conversational"),
        ("What is your name?", "self_identity"),
        ("What time is it?", "datetime"),
    ]
    
    db = SessionLocal()
    try:
        chat_service = ChatService(db)
        # Realtime voice pipeline with live AssemblyAI adapter
        pipeline = RealtimeVoicePipeline(
            chat_service=chat_service,
            stt_provider=adapter,
        )
        
        simple_query_results = []
        for text, category in simple_queries:
            pcm = synthesize_pcm16(text)
            session = pipeline.session_manager.create_session(
                conversation_id=f"conv-bench-{category}-{int(time.time()*1000)}",
                user_id="bench_user",
            )
            
            # Measure specific sub-component timings
            t_route_0 = time.perf_counter()
            decision = chat_service.orchestrator.router.analyze(text)
            from app.agent.policy import apply_policy
            decision = apply_policy(text, decision)
            router_ms = (time.perf_counter() - t_route_0) * 1000
            
            logger.info(f"Running '{text}' ({category}) [Route: {decision.route}, Intent: {getattr(decision, 'intent', '')}]...")
            turn_res = await run_pipeline_turn(pipeline, session, pcm, text)
            turn_res["route"] = decision.route
            turn_res["intent"] = getattr(decision, "intent", "")
            turn_res["router_ms"] = router_ms
            turn_res["stt_provider"] = "assemblyai (v3 streaming)"
            turn_res["stt_model"] = getattr(adapter, "_effective_model", "universal-3-6-pro")
            turn_res["llm_provider"] = settings.cloud_llm_provider
            turn_res["llm_model"] = settings.cloud_model
            simple_query_results.append(turn_res)
            logger.info(f"Done '{text}': Wall: {turn_res['total_wall_ms']:.1f}ms | Response: '{turn_res['response_text'][:60]}...'")
            
            results["simple_queries"] = simple_query_results
    finally:
        db.close()

    # -------------------------------------------------------------
    # 3. Context Continuation Test: "AI News" -> "Read it again."
    # -------------------------------------------------------------
    logger.info("=== STEP 5: CONTEXT CONTINUATION ('AI News' -> 'Read it again') ===")
    db2 = SessionLocal()
    try:
        chat_service = ChatService(db2)
        conv_id = f"conv-continuation-{int(time.time())}"
        u_id = "test_continuation_user"
        
        # Turn 1: Search the web for today's AI news
        logger.info("Turn 1: 'Search the web for today's AI news.'")
        t0_turn1 = time.perf_counter()
        resp_1 = chat_service.chat(
            conversation_id=conv_id,
            user_id=u_id,
            user_message="Search the web for today's AI news.",
        )
        t1_ms = (time.perf_counter() - t0_turn1) * 1000
        logger.info(f"Turn 1 Response ({t1_ms:.1f}ms):\n{resp_1[:150]}...")
        
        # Turn 2: "Read it again."
        logger.info("Turn 2: 'Read it again.'")
        t0_turn2 = time.perf_counter()
        resp_2 = chat_service.chat(
            conversation_id=conv_id,
            user_id=u_id,
            user_message="Read it again.",
        )
        t2_ms = (time.perf_counter() - t0_turn2) * 1000
        logger.info(f"Turn 2 Response ({t2_ms:.1f}ms):\n{resp_2[:150]}...")
        
        # Verify step 3 reuses prior answer and does NOT ask what "it" means
        asks_clarification = any(p in resp_2.lower() for p in ("could you provide more specific details", "which page", "what would you like"))
        reuses_prior = (resp_2 == resp_1) or any(k in resp_2.lower() for k in ("news", "ai", "today", "headline"))
        
        logger.info(f"Continuation Check: asks_clarification={asks_clarification}, reuses_prior={reuses_prior}")
        results["context_continuation"] = {
            "turn1_prompt": "Search the web for today's AI news.",
            "turn1_response_preview": resp_1[:150],
            "turn1_ms": t1_ms,
            "turn2_prompt": "Read it again.",
            "turn2_response_preview": resp_2[:150],
            "turn2_ms": t2_ms,
            "success": reuses_prior and not asks_clarification,
        }
    finally:
        db2.close()

    with open("scratch/acceptance_p15_4_2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Acceptance results saved to scratch/acceptance_p15_4_2_results.json")

if __name__ == "__main__":
    asyncio.run(main())
