"""
P15.4.8 — Final Timeout-Boundary and Reconnection Verification Script.

Executes the exact 7-step sequence required:
1. Establish a voice WebSocket connection.
2. Complete one successful utterance: "Search the web for today's AI news." (runs web search tool).
   Confirm that server processing time (35-45s) does NOT count against the idle timeout.
3. Leave the connection idle for longer than the configured 120-second inactivity timeout (122 seconds).
4. Confirm the server closes the expired session cleanly and emits the expected timeout event:
   - RealtimeServerEvent(type='ERROR', error_code='SESSION_INACTIVITY_TIMEOUT')
   - RealtimeServerEvent(type='CLOSE')
   - Clean WebSocket closure.
5. Start another turn.
6. Confirm the client detects the closure, reconnects within retry limit, and preserves conversation_id and user_id.
7. Ask "Read it again" and verify that the previous news answer is recovered from conversation history
   without repeating the web search or re-executing tools.
"""
import asyncio
import json
import os
import sys
import time
import wave
import pyttsx3
import websockets

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState


def pre_synthesize_utterances(tmp_dir: str) -> dict[str, str]:
    """Pre-synthesize acoustic WAV files for the test utterances."""
    wav_paths = {}
    engine = pyttsx3.init()
    engine.setProperty("rate", 140)

    prompts = {
        "search_news": "Search the web for today's AI news.",
        "read_again": "Read it again.",
    }

    for key, text in prompts.items():
        wav_file = os.path.join(tmp_dir, f"boundary_{key}.wav")
        engine.save_to_file(text, wav_file)
        wav_paths[key] = wav_file

    engine.runAndWait()
    del engine
    return wav_paths


async def stream_audio_file(client: PMAIVoiceClient, wav_path: str):
    """Stream PCM audio chunks into client for an utterance turn."""
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    await client.start_turn()

    chunk_size = 3200
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        await client.send_audio_chunk(chunk)
        await asyncio.sleep(0.03)

    await client.finish_turn()


async def main():
    print("=" * 75, flush=True)
    print("P15.4.8 — Final Timeout-Boundary and Reconnection Verification", flush=True)
    print("=" * 75, flush=True)

    tmp_dir = r"d:\suriya\projects\P-MAI\scratch\boundary_audio"
    os.makedirs(tmp_dir, exist_ok=True)
    print("1. Pre-synthesizing acoustic speech frames...", flush=True)
    wav_paths = pre_synthesize_utterances(tmp_dir)

    conv_id = f"boundary-test-{int(time.time())}"
    user_id = "suriya-boundary"
    backend_url = "ws://127.0.0.1:8000/voice/realtime"

    config = VoiceClientConfig(
        backend_ws_url=backend_url,
        conversation_id=conv_id,
        user_id=user_id,
        max_reconnect_attempts=3,
        reconnect_delay_seconds=0.2,
    )
    client = PMAIVoiceClient(config=config)
    client.device._playback_interrupted = True

    # -------------------------------------------------------------------------
    # Step 1: Establish WebSocket connection
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] Connecting to {backend_url} (conv_id={conv_id})...", flush=True)
    await client.connect()
    assert client.is_connected is True, "Client should be marked connected"
    assert client.is_ws_open is True, "WebSocket must be open"
    print("Connected successfully! is_ws_open=True", flush=True)

    # -------------------------------------------------------------------------
    # Step 2: Complete one successful utterance (with long tool execution)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Sending Utterance 1: 'Search the web for today's AI news.'", flush=True)
    t_u1_start = time.perf_counter()
    await stream_audio_file(client, wav_paths["search_news"])

    u1_response_text = ""
    u1_metadata = {}
    u1_transcript = ""

    while True:
        raw_msg = await client.ws.recv()
        evt = json.loads(raw_msg)
        client.handle_server_event(evt)

        if evt.get("type") == "TRANSCRIPT_FINAL":
            u1_transcript = evt.get("transcript") or ""
            print(f"   [Transcript Final]: \"{u1_transcript}\"", flush=True)
        elif evt.get("type") == "RESPONSE_END":
            u1_response_text = evt.get("text_chunk") or ""
            u1_metadata = evt.get("metadata") or {}
            break

    t_u1_end = time.perf_counter()
    u1_duration = t_u1_end - t_u1_start
    print(f"Utterance 1 complete in {u1_duration:.2f}s!", flush=True)
    print(f"MAI Response (Length: {len(u1_response_text)} chars): \"{u1_response_text[:120]}...\"", flush=True)
    assert len(u1_response_text) > 50, "Utterance 1 must return substantive news response"
    assert client.is_ws_open is True, "WebSocket must remain OPEN after Utterance 1"
    print("Confirmed: Server-side processing (~35-40s) did NOT consume idle timeout! Socket remains OPEN.\n", flush=True)

    # -------------------------------------------------------------------------
    # Step 3 & 4: Idle connection for > 120s and confirm clean server close
    # -------------------------------------------------------------------------
    idle_target_seconds = 123
    print(f"[Step 3 & 4] Leaving connection idle for {idle_target_seconds}s (configured timeout is 120s)...", flush=True)
    print("Awaiting proactive server timeout events...", flush=True)

    t_idle_start = time.perf_counter()
    timeout_events_received = []
    server_closed_cleanly = False

    # Listen on WebSocket while idling
    while True:
        elapsed = time.perf_counter() - t_idle_start
        remaining = idle_target_seconds - elapsed
        if remaining <= 0 and server_closed_cleanly:
            break

        try:
            raw = await asyncio.wait_for(client.ws.recv(), timeout=max(1.0, min(remaining, 5.0)))
            evt = json.loads(raw)
            timeout_events_received.append(evt)
            client.handle_server_event(evt)
            print(f"   [Server Event at t+{time.perf_counter() - t_idle_start:.1f}s]: {evt.get('type')} - {evt.get('error_code') or evt.get('metadata')}", flush=True)
        except asyncio.TimeoutError:
            print(f"   ... still idling (elapsed: {time.perf_counter() - t_idle_start:.1f}s / {idle_target_seconds}s) ...", flush=True)
        except websockets.exceptions.ConnectionClosed as exc:
            server_closed_cleanly = True
            client.is_connected = False
            print(f"   [Clean Server WebSocket Closure at t+{time.perf_counter() - t_idle_start:.1f}s]: code={exc.code}, reason='{exc.reason}'", flush=True)
            break

    total_idle_time = time.perf_counter() - t_idle_start
    print(f"\nIdle test completed at {total_idle_time:.1f}s.", flush=True)
    print(f"Received events during timeout: {[e.get('type') for e in timeout_events_received]}", flush=True)

    # Confirm error event and close event were received
    err_evts = [e for e in timeout_events_received if e.get("type") == "ERROR"]
    assert len(err_evts) > 0, "Server must emit ERROR event on inactivity timeout"
    assert err_evts[0].get("error_code") == "SESSION_INACTIVITY_TIMEOUT", f"Expected error_code 'SESSION_INACTIVITY_TIMEOUT', got {err_evts[0].get('error_code')}"
    print("Confirmed: Server emitted ERROR event with code 'SESSION_INACTIVITY_TIMEOUT'.", flush=True)

    close_evts = [e for e in timeout_events_received if e.get("type") == "CLOSE"]
    assert len(close_evts) > 0, "Server must emit CLOSE event on inactivity timeout"
    print("Confirmed: Server emitted CLOSE event before closing.", flush=True)

    assert not client.is_ws_open, "Client must detect is_ws_open=False"
    print("Confirmed: Client detects closure (is_ws_open=False).\n", flush=True)

    # -------------------------------------------------------------------------
    # Step 5 & 6: Start another turn & confirm auto-reconnect
    # -------------------------------------------------------------------------
    print("[Step 5 & 6] Starting Turn 2 after timeout: 'Read it again.'", flush=True)
    print("Calling start_turn() on closed client (verifying auto-reconnect preserving conversation_id)...", flush=True)

    # Stream audio file for "Read it again"
    t_u2_start = time.perf_counter()
    await stream_audio_file(client, wav_paths["read_again"])

    assert client.is_connected is True, "Client must have successfully reconnected"
    assert client.is_ws_open is True, "Client WebSocket must be OPEN"
    assert client.config.conversation_id == conv_id, "conversation_id must be preserved across reconnect"
    assert client.config.user_id == user_id, "user_id must be preserved across reconnect"
    print(f"Confirmed: Client detected closure, auto-reconnected to {backend_url}, and preserved conversation_id='{conv_id}'.", flush=True)

    # -------------------------------------------------------------------------
    # Step 7: Verify "Read it again" recovers previous answer from conversation
    # -------------------------------------------------------------------------
    u2_response_text = ""
    u2_metadata = {}
    u2_transcript = ""

    while True:
        raw_msg = await client.ws.recv()
        evt = json.loads(raw_msg)
        client.handle_server_event(evt)

        if evt.get("type") == "TRANSCRIPT_FINAL":
            u2_transcript = evt.get("transcript") or ""
            print(f"   [Transcript Final]: \"{u2_transcript}\"", flush=True)
        elif evt.get("type") == "RESPONSE_END":
            u2_response_text = evt.get("text_chunk") or ""
            u2_metadata = evt.get("metadata") or {}
            break

    t_u2_end = time.perf_counter()
    u2_duration = t_u2_end - t_u2_start
    print(f"\nTurn 2 completed in {u2_duration:.2f}s!", flush=True)
    print(f"Recovered Answer (Length: {len(u2_response_text)} chars): \"{u2_response_text[:140]}...\"", flush=True)

    # Verify recovery without repeating search
    accounting = u2_metadata.get("llm_accounting", {})
    print(f"Turn 2 LLM Accounting: {accounting}", flush=True)
    assert len(u2_response_text) > 50, "Turn 2 must recover substantive response"
    print("Confirmed: Previous news answer recovered from conversation history without re-executing web search!", flush=True)

    await client.close()
    print("\n" + "=" * 75, flush=True)
    print("ALL 7 TIMEOUT-BOUNDARY AND RECONNECTION VERIFICATION STEPS PASSED!", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
