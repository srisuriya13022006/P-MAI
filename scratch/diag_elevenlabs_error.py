import asyncio
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

from app.voice.config import VoiceConfig
from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.resilient_adapter import ResilientTTSAdapter


async def live_smoke_test():
    cfg = VoiceConfig.from_settings()
    print("--- 1. Testing VoiceConfig resolution ---")
    print(f"tts_provider: {cfg.tts_provider}")
    print(f"default_voice: {cfg.default_voice}")
    print(f"elevenlabs_voice_id: {cfg.elevenlabs_voice_id}")
    print(f"elevenlabs_model_id: {cfg.elevenlabs_model_id}")

    # Initialize adapter with configured credentials
    primary = ElevenLabsTTSAdapter(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
        model_id=cfg.elevenlabs_model_id,
    )
    fallback = LocalTTSAdapter()
    resilient = ResilientTTSAdapter(
        primary_provider=primary,
        fallback_provider=fallback,
        fallback_enabled=True,
    )

    print("\n--- 2. Live Synthesis with configured voice ---")
    res = await resilient.synthesize("Hello! This is P-MAI voice diagnostic test.")
    print(f"Synthesize SUCCESS: provider='{res.provider}', audio_bytes={len(res.audio_bytes)}, duration={res.duration_seconds:.2f}s")
    assert res.provider == "elevenlabs"
    assert len(res.audio_bytes) > 0

    print("\n--- 3. Live Streaming with voice='natural-en-1' (legacy default fallback test) ---")
    # This was the exact call that previously crashed with 400 invalid_uid
    chunks = []
    total_bytes = 0
    async for chunk in resilient.synthesize_stream("Testing voice override resolution.", voice="natural-en-1"):
        chunks.append(chunk)
        total_bytes += len(chunk.audio_bytes)
    print(f"Streaming SUCCESS: provider='{chunks[0].provider}', chunks={len(chunks)}, total_bytes={total_bytes}")
    assert chunks[0].provider == "elevenlabs"
    assert total_bytes > 0

    print("\n=== All live smoke tests PASSED ===")


if __name__ == "__main__":
    asyncio.run(live_smoke_test())
