"""
Live smoke verification of ElevenLabs TTS integration contract.
Synthesizes short emotional sentence and verifies audio properties, output format,
and playback through desktop AudioDevice.
Strictly prevents logging or printing secrets.
"""
import asyncio
import io
import time
import wave
import os
from dotenv import load_dotenv

load_dotenv()

from app.voice.config import VoiceConfig
from app.voice.audio import is_wav_container
from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from client.voice.audio_device import AudioDevice


async def main():
    print("=" * 70)
    print("P15.5 — Live ElevenLabs Contract Verification Smoke Test")
    print("=" * 70)

    cfg = VoiceConfig.from_settings()
    has_key = bool(cfg.elevenlabs_api_key)
    print(f"Backend API Key Configured: {has_key}")
    print(f"Configured Voice ID: {cfg.elevenlabs_voice_id}")
    print(f"Configured Model ID: {cfg.elevenlabs_model_id}")

    if not has_key:
        print("RESULT: LIVE TEST SKIPPED (No API key found)")
        return

    adapter = ElevenLabsTTSAdapter(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
        model_id=cfg.elevenlabs_model_id,
        stability=cfg.elevenlabs_stability,
        similarity_boost=cfg.elevenlabs_similarity_boost,
        output_format="pcm_16000",
    )

    prompt = "[excited] Great work! Your tests are passing."
    print(f"\n1. Synthesizing test prompt: '{prompt}'")
    t0 = time.perf_counter()

    result = await adapter.synthesize(prompt)
    duration_call = round(time.perf_counter() - t0, 3)

    print(f"   API Response received in: {duration_call}s")
    print(f"   Audio bytes returned: {len(result.audio_bytes)}")
    print(f"   Content Type: {result.content_type}")
    print(f"   Provider: {result.provider}")

    assert len(result.audio_bytes) > 0, "No audio bytes returned"
    assert is_wav_container(result.audio_bytes), "Audio is not a valid WAV container"

    with wave.open(io.BytesIO(result.audio_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration_audio = round(n_frames / framerate, 2)

        print("\n2. Audio Format Verification:")
        print(f"   Channels: {n_channels} (Expected: 1 - Mono)")
        print(f"   Sample Width: {sampwidth * 8}-bit (Expected: 16-bit)")
        print(f"   Sample Rate: {framerate} Hz (Expected: 16000 Hz)")
        print(f"   Audio Duration: {duration_audio}s")

        assert n_channels == 1, f"Expected 1 channel, got {n_channels}"
        assert sampwidth == 2, f"Expected 2 bytes per sample, got {sampwidth}"
        assert framerate == 16000, f"Expected 16000 Hz, got {framerate}"

    print("\n3. Streaming Verification (synthesize_stream):")
    stream_chunks = []
    t_first_chunk = None
    async for chunk in adapter.synthesize_stream("Quick streaming verification."):
        if t_first_chunk is None:
            t_first_chunk = round(time.perf_counter() - t0, 3)
        stream_chunks.append(chunk)

    print(f"   Streamed chunks received: {len(stream_chunks)}")
    print(f"   Total streaming bytes: {sum(len(c.audio_bytes) for c in stream_chunks)}")
    assert len(stream_chunks) > 0, "No stream chunks received"
    for c in stream_chunks:
        assert len(c.audio_bytes) % 2 == 0, "Unaligned PCM chunk received"

    print("\n4. Client Audio Playback Verification:")
    device = AudioDevice(sample_rate=16000, channels=1)
    device.check_availability()
    # Play the synthesized audio
    played = device.play_audio_chunk(result.audio_bytes)
    print(f"   AudioDevice.play_audio_chunk playback result: {played}")
    assert played is True, "Audio playback failed"

    print("\n" + "=" * 70)
    print("ALL LIVE ELEVENLABS CONTRACT VERIFICATIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
