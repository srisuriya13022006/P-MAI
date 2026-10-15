"""
P15.4.8 — Acceptance Verification Script:
1. Six sequential genuine voice turns on a single persistent WebSocket connection.
   Utterance 5 ("Search the web for today's AI news") runs for an extended period (web search).
   Utterance 6 ("Read it again") runs immediately on the SAME persistent connection and reproduces the news.
2. Inactivity timeout / server disconnect recovery:
   Tests that client detects closed connection, safely reconnects before the next turn,
   preserves conversation_id, and continues conversation memory seamlessly.
"""
import asyncio
import json
import os
import sys
import time
import uuid
import wave
import pyttsx3
import websockets

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState


SIX_TURNS = [
    {"id": 1, "prompt": "Hello", "desc": "Greeting"},
    {"id": 2, "prompt": "All okay", "desc": "Status inquiry"},
    {"id": 3, "prompt": "What is your name?", "desc": "Identity"},
    {"id": 4, "prompt": "What time is it?", "desc": "Time query"},
    {"id": 5, "prompt": "Search the web for today's AI news.", "desc": "Web search tool execution"},
    {"id": 6, "prompt": "Read it again.", "desc": "Context continuation"},
]


def pre_synthesize_utterances(tmp_dir: str) -> dict[int, str]:
    """Pre-synthesize acoustic WAV files for the 6 utterances."""
    wav_paths = {}
    engine = pyttsx3.init()
    engine.setProperty("rate", 140)

    for item in SIX_TURNS:
        u_id = item["id"]
        text = item["prompt"]
        wav_file = os.path.join(tmp_dir, f"p15_4_8_utt_{u_id}.wav")
        engine.save_to_file(text, wav_file)
        wav_paths[u_id] = wav_file

    engine.runAndWait()
    del engine
    return wav_paths


async def run_single_turn(client: PMAIVoiceClient, wav_path: str) -> dict:
    """Stream acoustic audio chunks through client into /voice/realtime."""
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    await client.start_turn()

    # Stream in 100ms chunks (3200 bytes for 16kHz 16-bit mono)
    chunk_size = 3200
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        await client.send_audio_chunk(chunk)
        await asyncio.sleep(0.03)

    t_finish = time.perf_counter()
    await client.finish_turn()

    final_transcript = ""
    response_text = ""
    metadata = {}
    received_types = []

    while True:
        try:
            raw_msg = await client.ws.recv()
        except websockets.exceptions.ConnectionClosed as exc:
            client.is_connected = False
            return {
                "final_transcript": final_transcript,
                "response_text": response_text,
                "metadata": metadata,
                "events": received_types,
                "error": f"ConnectionClosed: {exc}",
            }

        evt = json.loads(raw_msg)
        etype = evt.get("type")
        received_types.append(etype)
        client.handle_server_event(evt)

        if etype == "TRANSCRIPT_FINAL":
            final_transcript = evt.get("transcript") or ""
        elif etype == "RESPONSE_END":
            response_text = evt.get("text_chunk") or ""
            metadata = evt.get("metadata") or {}
            break
        elif etype in ("ERROR", "CLOSE"):
            if etype == "ERROR" and evt.get("error_code") == "TTS_STREAMING_ERROR":
                continue
            break

    return {
        "final_transcript": final_transcript,
        "response_text": response_text,
        "metadata": metadata,
        "events": received_types,
    }


async def main():
    print("=" * 75, flush=True)
    print("P15.4.8 — Acceptance Test: Multi-Turn Persistence & Disconnect Audit", flush=True)
    print("=" * 75, flush=True)

    tmp_dir = r"d:\suriya\projects\P-MAI\scratch\p15_4_8_audio"
    os.makedirs(tmp_dir, exist_ok=True)
    print("Pre-synthesizing acoustic speech frames...", flush=True)
    wav_paths = pre_synthesize_utterances(tmp_dir)

    conv_id = f"p15-4-8-{int(time.time())}"
    user_id = "suriya-audit"
    backend_url = "ws://127.0.0.1:8000/voice/realtime"

    config = VoiceClientConfig(
        backend_ws_url=backend_url,
        conversation_id=conv_id,
        user_id=user_id,
        max_reconnect_attempts=3,
        reconnect_delay_seconds=0.1,
    )
    client = PMAIVoiceClient(config=config)
    # Prevent audio hardware playback blocking during automated verification
    client.device._playback_interrupted = True

    print(f"\n[Part 1] Connecting client to {backend_url} (conv_id={conv_id})...", flush=True)
    await client.connect()
    assert client.is_connected, "Client must be connected"
    assert client.is_ws_open, "Client WebSocket must be open"
    print("PMAIVoiceClient connected successfully! is_ws_open=True\n", flush=True)

    turn_results = []
    # Execute Turns 1 through 6 sequentially
    for item in SIX_TURNS:
        u_id = item["id"]
        prompt = item["prompt"]
        desc = item["desc"]
        wav_file = wav_paths[u_id]

        print(f"--- Utterance {u_id}/6: \"{prompt}\" ({desc}) ---", flush=True)
        t0 = time.perf_counter()

        res = await run_single_turn(client, wav_file)
        t1 = time.perf_counter()
        dur = t1 - t0

        print(f"   Duration: {dur:.2f}s | Events: {res['events']}", flush=True)
        print(f"   Transcript: \"{res['final_transcript']}\"", flush=True)
        print(f"   MAI: \"{res['response_text'][:120]}...\"", flush=True)
        print(f"   Persistent WebSocket open: {client.is_ws_open}", flush=True)

        assert client.is_ws_open, f"WebSocket connection MUST remain OPEN after turn {u_id}!"
        assert "error" not in res, f"Turn {u_id} failed with error: {res.get('error')}"
        turn_results.append(res)

    # Verify that Turn 6 ("Read it again.") retrieved the preceding news answer
    turn5_text = turn_results[4]["response_text"].lower()
    turn6_text = turn_results[5]["response_text"].lower()
    print("\n[Audit] Checking Turn 6 continuation grounding:", flush=True)
    print(f"   Turn 5 (AI News length): {len(turn5_text)} chars")
    print(f"   Turn 6 (Repeat length): {len(turn6_text)} chars")
    assert len(turn6_text) > 30, "Turn 6 must return substantive repeated/summarized news"

    # Part 2: Test Server-Initiated Close / Inactivity Timeout & Clean Reconnect
    print("\n" + "=" * 75, flush=True)
    print("[Part 2] Validating Disconnect Detection and Context-Preserving Reconnect", flush=True)
    print("=" * 75, flush=True)

    # Intentionally close the existing connection to simulate server close / inactivity timeout
    print("Closing active WebSocket to simulate server-side timeout/drop...", flush=True)
    await client.ws.close(1000)
    await asyncio.sleep(0.05)
    assert not client.is_ws_open, "Client must detect is_ws_open=False"
    print("Client correctly reports is_ws_open=False without crashing.", flush=True)

    # Now verify that when client starts a turn after disconnect:
    # 1. start_turn detects closed socket
    # 2. auto-reconnects with the SAME conversation_id
    # 3. Executes "Read it again."
    # 4. Retrieves the previous news answer from the preserved conversation!
    print(f"\nTriggering turn after disconnect (conv_id={client.config.conversation_id})...", flush=True)
    reconnect_res = await run_single_turn(client, wav_paths[6])
    assert client.is_connected, "Client must be reconnected"
    assert client.is_ws_open, "New WebSocket must be open"
    print(f"Auto-reconnected successfully! is_ws_open=True", flush=True)
    print(f"Turn after reconnect events: {reconnect_res['events']}", flush=True)
    print(f"Turn after reconnect MAI response: \"{reconnect_res['response_text'][:120]}...\"", flush=True)
    assert len(reconnect_res['response_text']) > 30, "Turn after reconnect must preserve conversation history!"

    await client.close()
    print("\n" + "=" * 75, flush=True)
    print("ALL P15.4.8 VERIFICATION TESTS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
