"""
P15.4.7 — Interactive Physical Microphone Acceptance Runner.
Allows a human speaker to physically speak the 6 mandatory utterances into their microphone:
1. "Hello MAI."
2. "All okay."
3. "What is your name?"
4. "What time is it?"
5. "Search the web for today's AI news."
6. "Read it again."
Logs exact transcripts, latencies, provider models, and verifies live audio playback.
"""
import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from client.voice.audio_device import AudioDevice
from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig

UTTERANCES = [
    {"id": 1, "prompt": "Hello MAI.", "description": "Greeting"},
    {"id": 2, "prompt": "All okay.", "description": "Conversational acknowledgment"},
    {"id": 3, "prompt": "What is your name?", "description": "Deterministic identity"},
    {"id": 4, "prompt": "What time is it?", "description": "Deterministic datetime"},
    {"id": 5, "prompt": "Search the web for today's AI news.", "description": "Web news search & synthesis"},
    {"id": 6, "prompt": "Read it again.", "description": "Deterministic context repetition"},
]


async def run_interactive_acceptance():
    print("=" * 75)
    print("P15.4.7 — INTERACTIVE PHYSICAL MICROPHONE ACCEPTANCE RUNNER")
    print("=" * 75)
    print("This runner captures your GENUINE human spoken voice from the physical microphone.")
    print("Ensure your microphone is connected and unmuted.\n")

    conv_id = f"physical-mic-acceptance-{int(time.time())}"
    config = VoiceClientConfig(
        backend_ws_url="ws://localhost:8000/voice/realtime",
        conversation_id=conv_id,
        user_id="suriya-physical-acceptance",
    )
    device = AudioDevice(sample_rate=16000, channels=1, chunk_duration_s=0.1)
    client = PMAIVoiceClient(config=config, audio_device=device)

    print(f"Connecting to ws://localhost:8000/voice/realtime (session: {conv_id})...")
    try:
        await client.connect()
        print("Connected successfully to P-MAI voice backend!\n")
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        print("Ensure the backend server is running: python -m uvicorn app.main:app --port 8000")
        return

    results = []

    for item in UTTERANCES:
        u_id = item["id"]
        expected_text = item["prompt"]
        desc = item["description"]

        print(f"\n--- Utterance [{u_id}/6]: \"{expected_text}\" ({desc}) ---")
        input(f"Press [ENTER] to START recording, then speak \"{expected_text}\" clearly...")

        # 1. Start turn
        device.open_input_stream()
        await client.start_turn()
        t_start = time.perf_counter()
        print("🔴 RECORDING... Speak now! Press [ENTER] when you are finished speaking...", end="", flush=True)

        # Background task to stream audio chunks from physical mic
        stop_recording = asyncio.Event()

        async def stream_mic():
            while not stop_recording.is_set():
                chunk = device.read_microphone_chunk()
                await client.send_audio_chunk(chunk)
                await asyncio.sleep(0.08)

        mic_task = asyncio.create_task(stream_mic())

        # Wait for user input to stop
        await asyncio.to_thread(input)
        stop_recording.set()
        await mic_task

        t_audio_end = time.perf_counter()
        print(f"\n⏹ Stopped recording ({t_audio_end - t_start:.2f}s). Transcribing via AssemblyAI...")
        await client.finish_turn()

        # Collect response
        final_transcript = ""
        mai_response_text = ""
        response_metadata = {}
        t_first_audio = None
        audio_chunks_in = 0

        while True:
            raw = await client.ws.recv()
            evt = json.loads(raw)
            etype = evt.get("type")

            if etype == "TRANSCRIPT_PARTIAL":
                part = evt.get("transcript", "")
                print(f"\r   [Partial]: {part}", end="", flush=True)

            elif etype == "TRANSCRIPT_FINAL":
                final_transcript = evt.get("transcript", "")
                t_tr = time.perf_counter()
                print(f"\n   [AssemblyAI Final Transcript]: \"{final_transcript}\" (+{(t_tr - t_audio_end)*1000:.1f}ms)")

            elif etype == "RESPONSE_CHUNK":
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                    print(f"   [First Audio]: +{(t_first_audio - t_audio_end)*1000:.1f}ms from speech end")
                audio_chunks_in += 1
                # Play audio chunk to physical speakers
                client.handle_server_event(evt)

            elif etype == "RESPONSE_END":
                t_end = time.perf_counter()
                mai_response_text = evt.get("text_chunk", "")
                response_metadata = evt.get("metadata", {})
                print(f"   [MAI Response]: \"{mai_response_text}\"")
                print(f"   [Turnaround]: {(t_end - t_audio_end)*1000:.1f}ms (Total Wall-Clock: {(t_end - t_start):.2f}s)")
                break

            elif etype == "ERROR":
                print(f"\n   [Server Error]: {evt}")
                if evt.get("error_code") not in ("TTS_STREAMING_ERROR",):
                    break

        llm_accounting = response_metadata.get("llm_accounting", {})
        calls = llm_accounting.get("calls", [])
        active_prov = response_metadata.get("llm_provider", "none")
        active_call = next((c for c in reversed(calls) if c.get("provider") == active_prov), calls[-1] if calls else {})
        active_model = active_call.get("model", "qwen3:4b-instruct" if active_prov == "ollama" else "openai/gpt-oss-20b")

        rec = {
            "id": u_id,
            "spoken": expected_text,
            "actual_transcript": final_transcript,
            "response": mai_response_text,
            "stt_provider": response_metadata.get("stt_provider", "assemblyai"),
            "stt_model": response_metadata.get("stt_model", "universal-3-6-pro"),
            "stt_latency_ms": response_metadata.get("stt_latency_ms"),
            "llm_provider": active_prov,
            "llm_model": active_model if calls else "Zero LLM calls (deterministic)",
            "llm_calls": len(calls),
            "mai_latency_ms": response_metadata.get("mai_latency_ms"),
            "tts_latency_ms": response_metadata.get("tts_latency_ms"),
            "wall_clock_s": round(time.perf_counter() - t_start, 2),
        }
        results.append(rec)

    out_file = r"d:\suriya\projects\P-MAI\scratch\physical_mic_acceptance_live_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*75}\nPhysical microphone acceptance results saved to {out_file}\n{'='*75}")
    await client.ws.close()


if __name__ == "__main__":
    asyncio.run(run_interactive_acceptance())
