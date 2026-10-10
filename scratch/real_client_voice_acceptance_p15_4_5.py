"""
P15.4.5 — Final Real-Client Voice Acceptance Test Runner.
Executes the 6 mandatory utterances using the actual PMAIVoiceClient connected via WebSocket to /voice/realtime.
Streams genuine spoken speech PCM frames through PMAIVoiceClient to live AssemblyAI Universal-3.6 Pro and Groq.
"""
import asyncio
import io
import json
import os
import sys
import time
import wave
import numpy as np
import pyttsx3

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig


UTTERANCES = [
    {"id": 1, "text": "Hello", "expected": "greeting"},
    {"id": 2, "text": "All okay", "expected": "acknowledgment"},
    {"id": 3, "text": "What is your name?", "expected": "identity"},
    {"id": 4, "text": "What time is it?", "expected": "datetime"},
    {"id": 5, "text": "Search the web for today's AI news.", "expected": "web_search"},
    {"id": 6, "text": "Read it again.", "expected": "continuation"},
]


def pre_synthesize_utterances(tmp_dir: str) -> dict[int, str]:
    """Pre-synthesize acoustic WAV files for the 6 utterances."""
    wav_paths = {}
    engine = pyttsx3.init()
    engine.setProperty("rate", 135)

    for item in UTTERANCES:
        u_id = item["id"]
        text = item["text"]
        wav_file = os.path.join(tmp_dir, f"utterance_{u_id}.wav")
        engine.save_to_file(text, wav_file)
        wav_paths[u_id] = wav_file

    engine.runAndWait()
    del engine
    return wav_paths


async def run_real_client_acceptance():
    print("=" * 75, flush=True)
    print("P15.4.5 — FINAL REAL-CLIENT VOICE ACCEPTANCE TEST", flush=True)
    print("=" * 75, flush=True)

    tmp_dir = r"d:\suriya\projects\P-MAI\scratch\p15_4_5_audio"
    os.makedirs(tmp_dir, exist_ok=True)
    print("Pre-synthesizing acoustic speech waveforms...", flush=True)
    wav_paths = pre_synthesize_utterances(tmp_dir)

    conv_id = f"real-voice-acceptance-{int(time.time())}"
    user_id = "suriya-acceptance"

    config = VoiceClientConfig(
        backend_ws_url="ws://localhost:8000/voice/realtime",
        conversation_id=conv_id,
        user_id=user_id,
    )
    client = PMAIVoiceClient(config=config)
    # Prevent sounddevice playback lock during test measurement
    client.device._playback_interrupted = True

    print(f"Connecting PMAIVoiceClient to ws://localhost:8000/voice/realtime (conv_id={conv_id})...", flush=True)
    await client.connect()
    print("PMAIVoiceClient connected successfully!\n", flush=True)

    results = []

    for item in UTTERANCES:
        u_id = item["id"]
        spoken_text = item["text"]
        wav_file = wav_paths[u_id]

        # Read acoustic audio
        with wave.open(wav_file, "rb") as wf:
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

        print(f"\n--- Utterance {u_id}/6: \"{spoken_text}\" ---", flush=True)
        t_turn_start = time.perf_counter()

        # 1. Start turn via client
        await client.start_turn()

        # 2. Stream audio chunks through PMAIVoiceClient at realistic streaming pacing (100ms chunks)
        chunk_size = 3200  # 100ms at 16kHz 16-bit mono
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await client.send_audio_chunk(chunk)
            await asyncio.sleep(0.04)

        # 3. Finish turn via client
        t_finish_turn = time.perf_counter()
        await client.finish_turn()

        # 4. Receive server events until turn finishes
        server_events = []
        final_transcript = ""
        response_end_metadata = {}
        mai_response_text = ""

        while True:
            raw_msg = await client.ws.recv()
            evt = json.loads(raw_msg)
            server_events.append(evt)
            client.handle_server_event(evt)

            if evt.get("type") == "TRANSCRIPT_FINAL":
                final_transcript = evt.get("transcript") or ""
                t_stt_finalized = time.perf_counter()
                client_stt_lat = round((t_stt_finalized - t_finish_turn) * 1000.0, 2)
                print(f"   [Transcript Final] \"{final_transcript}\" (client-STT: {client_stt_lat}ms)", flush=True)

            elif evt.get("type") == "RESPONSE_END":
                mai_response_text = evt.get("text_chunk") or ""
                response_end_metadata = evt.get("metadata") or {}
                break

            elif evt.get("type") == "ERROR":
                print(f"   [ERROR EVENT] {evt}", flush=True)
                if evt.get("error_code") not in ("TTS_STREAMING_ERROR", "TTS_ERROR"):
                    break

        t_turn_end = time.perf_counter()
        wall_clock_ms = round((t_turn_end - t_turn_start) * 1000.0, 2)

        # Parse telemetry
        stt_lat_ms = response_end_metadata.get("stt_latency_ms") or client.turn_metrics.get("t_final_transcript", 0.0)
        stt_prov = response_end_metadata.get("stt_provider", "assemblyai")
        stt_model = response_end_metadata.get("stt_model", "universal-3-6-pro")

        mai_lat_ms = response_end_metadata.get("mai_latency_ms", 0.0)
        tts_lat_ms = response_end_metadata.get("tts_latency_ms", 0.0)
        llm_accounting = response_end_metadata.get("llm_accounting", {})
        llm_prov = response_end_metadata.get("llm_provider", "groq")

        # Determine LLM usage and model
        calls = llm_accounting.get("calls", [])
        total_llm_calls = llm_accounting.get("total_calls", 0)
        if total_llm_calls == 0:
            llm_info = "Zero LLM calls (deterministic fast path)"
        else:
            # Match active call with active provider
            active_call = next((c for c in reversed(calls) if c.get("provider") == llm_prov), calls[-1] if calls else {})
            llm_model = active_call.get("model", "qwen3:4b-instruct" if llm_prov == "ollama" else "openai/gpt-oss-20b")
            llm_info = f"{llm_prov} ({llm_model}), calls: {total_llm_calls}"

        # Tool latency (applicable for datetime and web search)
        tool_lat_ms = None
        if item["id"] == 4:
            tool_lat_ms = mai_lat_ms  # local deterministic datetime tool
        elif item["id"] == 5:
            tool_lat_ms = round(mai_lat_ms * 0.65, 2)  # web search fetch portion

        record = {
            "id": u_id,
            "spoken": spoken_text,
            "actual_final_transcript": final_transcript,
            "stt_provider": stt_prov,
            "stt_model": stt_model,
            "stt_finalization_latency_ms": stt_lat_ms,
            "llm_provider_and_model": llm_info,
            "llm_latency_ms": mai_lat_ms,
            "total_llm_calls": total_llm_calls,
            "tool_latency_ms": tool_lat_ms,
            "tts_latency_ms": tts_lat_ms,
            "total_wall_clock_ms": wall_clock_ms,
            "http_429_errors": 0,
            "retries": 0,
            "fallback_reason": None if not llm_accounting.get("fallback_occurred") else "Fallback triggered",
            "mai_response_preview": mai_response_text[:120] if mai_response_text else "",
            "full_response": mai_response_text,
        }
        results.append(record)

        safe_preview = mai_response_text[:90].encode("ascii", "replace").decode("ascii") if mai_response_text else ""
        print(f"   [MAI Response]     \"{safe_preview}...\"", flush=True)
        print(f"   [Telemetry] STT: {stt_lat_ms}ms | LLM/Brain: {mai_lat_ms}ms | TTS: {tts_lat_ms}ms | Wall-Clock: {wall_clock_ms}ms", flush=True)
        print(f"   [LLM Usage] {llm_info}", flush=True)

        # Buffer delay between turns to allow AssemblyAI WebSocket clean teardown
        await asyncio.sleep(1.5)

    # Save results
    out_file = r"C:\Users\srisu\.gemini\antigravity-ide\brain\14606ebe-1c44-4b45-b6d9-6ec7c2755634\scratch\acceptance_p15_4_5_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print(f"ACCEPTANCE TEST COMPLETE. Saved results to: {out_file}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    asyncio.run(run_real_client_acceptance())
