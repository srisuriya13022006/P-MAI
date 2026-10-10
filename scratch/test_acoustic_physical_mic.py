"""
Test physical acoustic capture:
Plays acoustic audio through physical speakers into the physical room,
while physical Microphone Array captures the sound waves from the air,
and streams the recorded microphone PCM into AssemblyAI live WebSocket.
"""
import asyncio
import base64
import json
import os
import sys
import time
import numpy as np
import sounddevice as sd
import websockets

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

from app.voice.audio import AudioNormalizer, strip_wav_header


async def test_acoustic_mic():
    print("=" * 70)
    print("PHYSICAL ACOUSTIC AIR-GAP MICROPHONE TEST")
    print("=" * 70)

    # 1. Prepare speech wave
    import pyttsx3
    wav_path = r"d:\suriya\projects\P-MAI\scratch\p15_4_6_diag\mic_prompt.wav"
    engine = pyttsx3.init()
    engine.setProperty("rate", 140)
    engine.save_to_file("Hello", wav_path + ".tmp.wav")
    engine.runAndWait()
    del engine

    with open(wav_path + ".tmp.wav", "rb") as f:
        raw_b = f.read()
    try:
        os.unlink(wav_path + ".tmp.wav")
    except OSError:
        pass

    normalizer = AudioNormalizer(16000)
    norm_wav = normalizer.normalize_wav(raw_b)
    pcm_play = strip_wav_header(norm_wav)
    samples_play = np.frombuffer(pcm_play, dtype=np.int16)

    # 2. Open physical microphone
    print("Opening physical Microphone Array (device 1)...")
    mic_stream = sd.RawInputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=1600,
    )
    mic_stream.start()

    # 3. Connect to live WebSocket
    ws_url = "ws://localhost:8000/voice/realtime?conversation_id=acoustic-mic-test&user_id=suriya"
    print(f"Connecting to {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "AUDIO_START"}))

        # Start playback on physical speakers in background thread
        print("Playing audio into room through physical speakers...")
        sd.play(samples_play, samplerate=16000, blocking=False)

        # Record physical acoustic wave from microphone
        print("Capturing live physical microphone audio...")
        recorded_chunks = []
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 2.5:
            rec_chunk = mic_stream.read(1600)[0]
            b64 = base64.b64encode(rec_chunk).decode("ascii")
            await ws.send(json.dumps({
                "type": "AUDIO_CHUNK",
                "audio_base64": b64,
                "content_type": "audio/wav",
            }))
            recorded_chunks.append(rec_chunk)
            await asyncio.sleep(0.04)

        t_end = time.perf_counter()
        print(f"Captured {len(recorded_chunks)} chunks from physical microphone ({len(recorded_chunks)*1600*2} bytes).")
        await ws.send(json.dumps({"type": "AUDIO_END"}))

        mic_stream.stop()
        mic_stream.close()

        final_transcript = ""
        response_text = ""
        audio_chunks_in = 0
        metadata = {}

        while True:
            msg = await ws.recv()
            evt = json.loads(msg)
            mtype = evt.get("type")
            if mtype == "TRANSCRIPT_FINAL":
                final_transcript = evt.get("transcript")
                print(f"  -> Physical Microphone Recognized: \"{final_transcript}\"")
            elif mtype == "RESPONSE_CHUNK":
                audio_chunks_in += 1
            elif mtype == "RESPONSE_END":
                response_text = evt.get("text_chunk")
                metadata = evt.get("metadata", {})
                print(f"  -> MAI Response: \"{response_text}\"")
                break
            elif mtype == "ERROR":
                print(f"  -> Event: {evt}")
                if evt.get("error_code") not in ("TTS_STREAMING_ERROR",):
                    break

        print("\nPhysical Microphone Verification Result:")
        print(f"  Transcript: {final_transcript}")
        print(f"  Response: {response_text}")
        print(f"  STT Latency: {metadata.get('stt_latency_ms')} ms")
        print(f"  Agent Latency: {metadata.get('mai_latency_ms')} ms")
        print(f"  TTS Latency: {metadata.get('tts_latency_ms')} ms")

        # Save result
        with open(r"d:\suriya\projects\P-MAI\scratch\acoustic_mic_result.json", "w", encoding="utf-8") as f:
            json.dump({
                "final_transcript": final_transcript,
                "response_text": response_text,
                "metadata": metadata,
            }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(test_acoustic_mic())
