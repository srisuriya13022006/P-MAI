"""
P15.4.6 — Desktop Physical Microphone Verification Harness.
Captures genuine physical audio from the machine's real microphone (Microphone Array Realtek),
streams to MAI WebSocket, transcribes via AssemblyAI Universal-3.6 Pro, executes agent reasoning,
synthesizes TTS, and verifies client speaker playback.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.audio_device import AudioDevice


async def run_desktop_mic_verification():
    print("=" * 75, flush=True)
    print("DESKTOP PHYSICAL MICROPHONE VERIFICATION TEST", flush=True)
    print("=" * 75, flush=True)

    conv_id = f"desktop-mic-{int(time.time())}"
    config = VoiceClientConfig(
        backend_ws_url="ws://localhost:8000/voice/realtime",
        conversation_id=conv_id,
        user_id="suriya-mic-test",
        sample_rate=16000,
        channels=1,
    )

    audio_device = AudioDevice(sample_rate=16000, channels=1, chunk_duration_s=0.1)
    audio_device.open_input_stream()
    print(f"Physical Microphone Status: is_active={audio_device._is_active}, input_stream={audio_device._input_stream}", flush=True)
    assert audio_device._is_active, "Physical microphone must be active!"

    client = PMAIVoiceClient(config=config, audio_device=audio_device)
    print("Connecting PMAIVoiceClient to live WebSocket ws://localhost:8000/voice/realtime...", flush=True)
    await client.connect()
    print("Connected successfully!\n", flush=True)

    t_audio_start = time.perf_counter()
    await client.start_turn()

    # Capture real physical microphone frames
    print(">>> Capturing live audio from Physical Microphone (3.0 seconds)...", flush=True)
    chunks_sent = 0
    t_first_chunk = None

    capture_duration = 3.0
    start_capture = time.perf_counter()
    while time.perf_counter() - start_capture < capture_duration:
        chunk = audio_device.read_microphone_chunk()
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
        await client.send_audio_chunk(chunk)
        chunks_sent += 1
        await asyncio.sleep(0.08)  # natural chunk interval

    t_audio_end = time.perf_counter()
    print(f">>> Finished live capture: {chunks_sent} physical PCM chunks streamed ({len(chunk)} bytes each).", flush=True)
    await client.finish_turn()

    # Collect server response
    final_transcript = ""
    response_text = ""
    audio_chunks_received = 0
    response_metadata = {}
    t_first_audio_recv = None

    while True:
        raw = await client.ws.recv()
        evt = json.loads(raw)
        etype = evt.get("type")

        if etype == "TRANSCRIPT_FINAL":
            final_transcript = evt.get("transcript", "")
            t_tr = time.perf_counter()
            print(f"  -> AssemblyAI Final Transcript: \"{final_transcript}\" (+{(t_tr - t_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)

        elif etype == "RESPONSE_CHUNK":
            if t_first_audio_recv is None:
                t_first_audio_recv = time.perf_counter()
                print(f"  -> First TTS Audio Chunk (+{(t_first_audio_recv - t_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)
            audio_chunks_received += 1
            client.handle_server_event(evt)

        elif etype == "RESPONSE_END":
            t_response_end = time.perf_counter()
            response_text = evt.get("text_chunk", "")
            response_metadata = evt.get("metadata", {})
            print(f"  -> RESPONSE_END (+{(t_response_end - t_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)
            break

        elif etype == "ERROR":
            print(f"  -> Server error event: {evt}", flush=True)
            if evt.get("error_code") not in ("TTS_STREAMING_ERROR",):
                break

    t_total_turn = time.perf_counter() - t_audio_start
    print("\n--- DESKTOP PHYSICAL MICROPHONE VERIFICATION METRICS ---", flush=True)
    print(f"  Physical Mic Stream Active: {audio_device._is_active}")
    print(f"  Chunks Sent: {chunks_sent}")
    print(f"  AssemblyAI Final Transcript: \"{final_transcript}\"")
    print(f"  Agent Response: \"{response_text[:80]}...\"")
    print(f"  Audio Chunks Received: {audio_chunks_received}")
    print(f"  STT Latency: {response_metadata.get('stt_latency_ms')} ms")
    print(f"  Agent Latency: {response_metadata.get('mai_latency_ms')} ms")
    print(f"  TTS Latency: {response_metadata.get('tts_latency_ms')} ms")
    print(f"  Total Turn Wall-Clock: {t_total_turn:.2f} s")

    results = {
        "physical_mic_active": audio_device._is_active,
        "chunks_sent": chunks_sent,
        "transcript": final_transcript,
        "response": response_text,
        "audio_chunks_received": audio_chunks_received,
        "stt_latency_ms": response_metadata.get("stt_latency_ms"),
        "mai_latency_ms": response_metadata.get("mai_latency_ms"),
        "tts_latency_ms": response_metadata.get("tts_latency_ms"),
        "total_wall_clock_s": round(t_total_turn, 2),
    }

    with open(r"d:\suriya\projects\P-MAI\scratch\desktop_mic_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    await client.ws.close()
    return results


if __name__ == "__main__":
    asyncio.run(run_desktop_mic_verification())
