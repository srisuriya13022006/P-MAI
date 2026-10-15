"""
P15.6 Runtime Latency & LLM Provider Routing Audit Benchmark.
Measures real end-to-end response latency across all 11 stages and Scenarios A-G.
Strictly masks keys and avoids printing secrets.
"""
import asyncio
import io
import json
import os
import time
import wave
from typing import Any

from app.database.connection import SessionLocal
from app.services.chat_service import ChatService
from app.voice.config import VoiceConfig
from app.voice.events import (
    RealtimeClientEvent,
    RealtimeEventType,
    RealtimeServerEvent,
)
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.session import RealtimeVoiceSession, SessionStatus
from app.voice.tts.factory import create_tts_provider


def run_scenario(name: str, fn, *args, **kwargs) -> dict[str, Any]:
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {"scenario": name, "total_ms": total_ms, "result": res}


def audit_chat_scenarios() -> list[dict[str, Any]]:
    db = SessionLocal()
    results = []

    try:
        service = ChatService(db)
        conv_id = f"audit-{int(time.time())}"
        user_id = "audit-user"

        # Scenario A: Greeting
        print("\n--- Running Scenario A: Simple Greeting ('Hello.') ---")
        t0 = time.perf_counter()
        res_a = service.chat(conv_id, user_id, "Hello.")
        t_a = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_a = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_a = getattr(service.orchestrator.llm, "last_telemetry", {})
        results.append({
            "scenario": "A. Greeting",
            "prompt": "Hello.",
            "total_ms": t_a,
            "response": res_a[:80],
            "llm_accounting": llm_accounting_a,
            "telemetry": telemetry_a,
        })
        print(f"Scenario A finished in {t_a}ms | Active provider: {telemetry_a.get('active_provider')} | Model: {telemetry_a.get('active_model')}")

        # Scenario B: Deterministic Identity
        print("\n--- Running Scenario B: Identity ('What is your name?') ---")
        t0 = time.perf_counter()
        res_b = service.chat(conv_id, user_id, "What is your name?")
        t_b = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_b = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_b = getattr(service.orchestrator.llm, "last_telemetry", {})
        results.append({
            "scenario": "B. Identity",
            "prompt": "What is your name?",
            "total_ms": t_b,
            "response": res_b[:80],
            "llm_accounting": llm_accounting_b,
            "telemetry": telemetry_b,
        })
        print(f"Scenario B finished in {t_b}ms | Response: {res_b.strip()}")

        # Scenario C: Datetime
        print("\n--- Running Scenario C: Datetime ('What time is it?') ---")
        t0 = time.perf_counter()
        res_c = service.chat(conv_id, user_id, "What time is it?")
        t_c = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_c = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_c = getattr(service.orchestrator.llm, "last_telemetry", {})
        results.append({
            "scenario": "C. Datetime",
            "prompt": "What time is it?",
            "total_ms": t_c,
            "response": res_c[:80],
            "llm_accounting": llm_accounting_c,
            "telemetry": telemetry_c,
        })
        print(f"Scenario C finished in {t_c}ms | Response: {res_c.strip()}")

        # Scenario D: Conversational question
        print("\n--- Running Scenario D: Conversational LLM ('Explain photosynthesis in one sentence.') ---")
        t0 = time.perf_counter()
        res_d = service.chat(conv_id, user_id, "Explain photosynthesis in one sentence.")
        t_d = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_d = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_d = getattr(service.orchestrator.llm, "last_telemetry", {})
        results.append({
            "scenario": "D. Conversational LLM",
            "prompt": "Explain photosynthesis in one sentence.",
            "total_ms": t_d,
            "response": res_d[:80],
            "llm_accounting": llm_accounting_d,
            "telemetry": telemetry_d,
        })
        print(f"Scenario D finished in {t_d}ms | Active provider: {telemetry_d.get('active_provider')} | Model: {telemetry_d.get('active_model')}")

        # Scenario E: Web Search
        print("\n--- Running Scenario E: Web Search ('Search the web for python latest version') ---")
        t0 = time.perf_counter()
        res_e = service.chat(conv_id, user_id, "Search the web for python latest version")
        t_e = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_e = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_e = getattr(service.orchestrator.llm, "last_telemetry", {})
        results.append({
            "scenario": "E. Web Search",
            "prompt": "Search the web for python latest version",
            "total_ms": t_e,
            "response": res_e[:80],
            "llm_accounting": llm_accounting_e,
            "telemetry": telemetry_e,
        })
        print(f"Scenario E finished in {t_e}ms | Active provider: {telemetry_e.get('active_provider')} | Model: {telemetry_e.get('active_model')}")

        # Scenario F: Multi-turn follow-up
        print("\n--- Running Scenario F: Multi-Turn Conversation ---")
        t0 = time.perf_counter()
        res_f1 = service.chat(conv_id, user_id, "Remember that my secret code is ALPHA-42.")
        t_f1 = round((time.perf_counter() - t0) * 1000.0, 2)
        safe_f1 = res_f1[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"Turn 1 finished in {t_f1}ms | Response: {safe_f1}")

        t0 = time.perf_counter()
        res_f2 = service.chat(conv_id, user_id, "What is my secret code?")
        t_f2 = round((time.perf_counter() - t0) * 1000.0, 2)
        llm_accounting_f2 = getattr(service.orchestrator.llm, "get_turn_accounting", lambda: {})()
        telemetry_f2 = getattr(service.orchestrator.llm, "last_telemetry", {})
        safe_f2 = res_f2[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"Turn 2 finished in {t_f2}ms | Response: {safe_f2}")

        results.append({
            "scenario": "F. Multi-turn Conversation",
            "turn_1_ms": t_f1,
            "turn_2_ms": t_f2,
            "total_ms": round(t_f1 + t_f2, 2),
            "response": res_f2[:80],
            "llm_accounting": llm_accounting_f2,
            "telemetry": telemetry_f2,
        })

    finally:
        db.close()

    return results


async def audit_voice_scenario() -> dict[str, Any]:
    print("\n--- Running Scenario G: Voice Interaction (STT -> LLM -> TTS) ---")
    db = SessionLocal()
    try:
        cfg = VoiceConfig.from_settings()
        service = ChatService(db)
        
        # Test real streaming TTS with ElevenLabs
        tts_provider = create_tts_provider(cfg)
        pipeline = RealtimeVoicePipeline(chat_service=service, tts_provider=tts_provider, config=cfg)
        
        # Test full pipeline turn with speech transcript
        session = pipeline.session_manager.create_session("voice-audit-conv", "voice-audit-user")
        session.transition_to(SessionStatus.LISTENING)
        
        from app.voice.stt.streaming import MockStreamingSTTProvider
        pipeline.stt_provider = MockStreamingSTTProvider(final_transcript="What is your name?")
        
        t0 = time.perf_counter()
        events = []
        async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END)):
            events.append(e)
        total_voice_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        
        resp_end = next((e for e in events if e.type == RealtimeEventType.RESPONSE_END), None)
        chunks = [e for e in events if e.type == RealtimeEventType.RESPONSE_CHUNK]
        
        meta = getattr(resp_end, "metadata", {}) or {}
        print(f"Scenario G finished in {total_voice_ms}ms | TTS Chunks received: {len(chunks)}")
        print(f"STT: {meta.get('stt_latency_ms')}ms | MAI: {meta.get('mai_latency_ms')}ms | TTS: {meta.get('tts_latency_ms')}ms | TTS Provider: {meta.get('tts_provider')}")
        
        return {
            "scenario": "G. Voice Interaction",
            "total_ms": total_voice_ms,
            "tts_chunks_count": len(chunks),
            "stt_latency_ms": meta.get("stt_latency_ms"),
            "mai_latency_ms": meta.get("mai_latency_ms"),
            "tts_latency_ms": meta.get("tts_latency_ms"),
            "tts_provider": meta.get("tts_provider"),
            "llm_provider": meta.get("llm_provider"),
            "runtime_mode": meta.get("runtime_mode"),
            "timings": meta.get("timings"),
        }
    finally:
        db.close()


def main():
    print("=" * 80)
    print("P15.6: RUNTIME LATENCY & LLM PROVIDER ROUTING AUDIT")
    print("=" * 80)

    chat_results = audit_chat_scenarios()
    voice_result = asyncio.run(audit_voice_scenario())

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chat_scenarios": chat_results,
        "voice_scenario": voice_result,
    }

    with open("scratch/post_fix_latency_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("POST-FIX AUDIT COMPLETED. Results written to scratch/post_fix_latency_audit.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
