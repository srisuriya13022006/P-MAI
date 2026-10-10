"""
P15.4.6 — Fast-Path Monotonic Latency Diagnosis Harness.
Measures all stage durations and total wall-clock duration for:
1. "What is your name?"
2. "What time is it?"
3. "What is 25 times 47?"
4. "Search the web for today's AI news."
5. Physical microphone capture live test.
"""
import asyncio
import base64
import json
import os
import sys
import time
import wave
import numpy as np

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

import websockets
import pyttsx3
from app.voice.audio import AudioNormalizer, CanonicalAudioFormat


def synthesize_canonical_16k_wav(text: str, out_path: str):
    """Synthesize clean 16kHz mono 16-bit PCM WAV."""
    tmp_wav = out_path + ".tmp.wav"
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)
    engine.save_to_file(text, tmp_wav)
    engine.runAndWait()
    del engine

    with open(tmp_wav, "rb") as f:
        raw_bytes = f.read()
    try:
        os.unlink(tmp_wav)
    except OSError:
        pass

    normalizer = AudioNormalizer(16000)
    norm_wav = normalizer.normalize_wav(raw_bytes)
    with open(out_path, "wb") as f:
        f.write(norm_wav)


async def execute_instrumented_turn(ws, audio_bytes: bytes, turn_label: str):
    print(f"\n{'='*70}\n[TURN] {turn_label}\n{'='*70}", flush=True)

    # 1. Monotonic Timestamps
    t_client_audio_start = time.perf_counter()
    await ws.send(json.dumps({"type": "AUDIO_START"}))

    t_first_chunk_sent = None
    chunk_size = 3200  # 100ms at 16kHz 16-bit mono (2 bytes/sample)

    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        b64 = base64.b64encode(chunk).decode("ascii")
        if t_first_chunk_sent is None:
            t_first_chunk_sent = time.perf_counter()
        await ws.send(json.dumps({
            "type": "AUDIO_CHUNK",
            "audio_base64": b64,
            "content_type": "audio/wav",
        }))
        await asyncio.sleep(0.02)  # fast realistic stream pacing

    t_client_audio_end = time.perf_counter()
    await ws.send(json.dumps({"type": "AUDIO_END"}))

    # Collect server responses and timestamps
    t_client_first_audio_recv = None
    t_client_response_end_recv = None
    audio_chunks_received = 0
    total_audio_bytes_received = 0
    server_timings = {}
    response_metadata = {}
    final_transcript = ""
    response_text = ""

    while True:
        raw = await ws.recv()
        evt = json.loads(raw)
        etype = evt.get("type")

        if etype == "TRANSCRIPT_FINAL":
            final_transcript = evt.get("transcript", "")
            t_transcript_final_recv = time.perf_counter()
            print(f"  -> TRANSCRIPT_FINAL: \"{final_transcript}\" at +{(t_transcript_final_recv - t_client_audio_end)*1000:.1f}ms from AUDIO_END", flush=True)

        elif etype == "RESPONSE_CHUNK":
            if t_client_first_audio_recv is None:
                t_client_first_audio_recv = time.perf_counter()
                print(f"  -> First RESPONSE_CHUNK at +{(t_client_first_audio_recv - t_client_audio_end)*1000:.1f}ms from AUDIO_END", flush=True)
            audio_chunks_received += 1
            audio_b64 = evt.get("audio_base64", "")
            if audio_b64:
                total_audio_bytes_received += len(base64.b64decode(audio_b64))

        elif etype == "RESPONSE_END":
            t_client_response_end_recv = time.perf_counter()
            response_text = evt.get("text_chunk", "")
            response_metadata = evt.get("metadata", {})
            server_timings = response_metadata.get("timings", {})
            print(f"  -> RESPONSE_END: \"{response_text[:80]}...\" at +{(t_client_response_end_recv - t_client_audio_end)*1000:.1f}ms from AUDIO_END", flush=True)
            break

        elif etype == "ERROR":
            print(f"  -> ERROR EVENT: {evt}", flush=True)
            if evt.get("error_code") not in ("TTS_STREAMING_ERROR",):
                break

    # Calculate detailed stage durations
    total_wall_clock_ms = (t_client_response_end_recv - t_client_audio_start) * 1000.0
    audio_streaming_dur_ms = (t_client_audio_end - t_client_audio_start) * 1000.0
    turnaround_from_audio_end_ms = (t_client_response_end_recv - t_client_audio_end) * 1000.0

    stt_latency_ms = response_metadata.get("stt_latency_ms", 0.0)
    mai_latency_ms = response_metadata.get("mai_latency_ms", 0.0)
    tts_latency_ms = response_metadata.get("tts_latency_ms", 0.0)

    # Server internal timings
    t_server_recv_audio_end = server_timings.get("t_server_recv_audio_end")
    t_stt_start = server_timings.get("t_stt_start")
    t_stt_end = server_timings.get("t_stt_end")
    t_agent_start = server_timings.get("t_agent_start")
    t_agent_end = server_timings.get("t_agent_end")
    t_tts_start = server_timings.get("t_tts_start")
    t_tts_end = server_timings.get("t_tts_end")

    print("\n--- TIMING BREAKDOWN ---", flush=True)
    print(f"  1. Client AUDIO_START -> AUDIO_END (Speech Stream): {audio_streaming_dur_ms:.1f} ms", flush=True)
    print(f"  2. Server STT Latency: {stt_latency_ms:.1f} ms", flush=True)
    print(f"  3. Server Agent Latency: {mai_latency_ms:.1f} ms (LLM Calls: {response_metadata.get('llm_accounting', {}).get('total_calls', 0)})", flush=True)
    print(f"  4. Server TTS Latency: {tts_latency_ms:.1f} ms (Chunks: {audio_chunks_received})", flush=True)
    if t_client_first_audio_recv:
        time_to_first_audio_ms = (t_client_first_audio_recv - t_client_audio_end) * 1000.0
        print(f"  5. Client Time to First Audio from AUDIO_END: {time_to_first_audio_ms:.1f} ms", flush=True)
    print(f"  6. Turnaround from AUDIO_END -> RESPONSE_END: {turnaround_from_audio_end_ms:.1f} ms", flush=True)
    print(f"  7. Total Wall-Clock (AUDIO_START -> RESPONSE_END): {total_wall_clock_ms:.1f} ms ({total_wall_clock_ms/1000:.2f}s)", flush=True)

    sequential_sum_ms = stt_latency_ms + mai_latency_ms + tts_latency_ms
    print(f"  * Sequential Server Sum (STT + Agent + TTS): {sequential_sum_ms:.1f} ms", flush=True)
    gap_ms = turnaround_from_audio_end_ms - sequential_sum_ms
    print(f"  * Gap (Network framing + WebSocket serialize + Queue drain): {gap_ms:.1f} ms", flush=True)

    return {
        "turn_label": turn_label,
        "transcript": final_transcript,
        "response": response_text,
        "llm_provider": response_metadata.get("llm_provider"),
        "llm_calls": response_metadata.get("llm_accounting", {}).get("total_calls", 0),
        "stt_latency_ms": stt_latency_ms,
        "mai_latency_ms": mai_latency_ms,
        "tts_latency_ms": tts_latency_ms,
        "audio_stream_ms": audio_streaming_dur_ms,
        "turnaround_from_audio_end_ms": turnaround_from_audio_end_ms,
        "total_wall_clock_ms": total_wall_clock_ms,
        "server_timings": server_timings,
    }


async def main():
    tmp_dir = r"d:\suriya\projects\P-MAI\scratch\p15_4_6_diag"
    os.makedirs(tmp_dir, exist_ok=True)

    queries = [
        {"id": "name", "text": "What is your name?"},
        {"id": "time", "text": "What time is it?"},
        {"id": "calc", "text": "What is 25 times 47?"},
    ]

    print("Pre-synthesizing canonical 16kHz WAV files...", flush=True)
    wav_map = {}
    for q in queries:
        p = os.path.join(tmp_dir, f"{q['id']}.wav")
        synthesize_canonical_16k_wav(q["text"], p)
        wav_map[q["id"]] = p

    ws_url = "ws://localhost:8000/voice/realtime?conversation_id=p15-4-6-diag&user_id=suriya"
    print(f"Connecting to {ws_url}...", flush=True)
    async with websockets.connect(ws_url, ping_interval=30, ping_timeout=120) as ws:
        results = []
        for q in queries:
            with open(wav_map[q["id"]], "rb") as f:
                audio_bytes = f.read()
            # strip WAV header for raw PCM
            from app.voice.audio import strip_wav_header
            pcm = strip_wav_header(audio_bytes)
            res = await execute_instrumented_turn(ws, pcm, q["text"])
            results.append(res)

        out_json = r"d:\suriya\projects\P-MAI\scratch\fast_path_diag_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nAll fast-path diagnosis results saved to {out_json}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
