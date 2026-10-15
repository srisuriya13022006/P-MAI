import asyncio
import time
from app.voice.config import VoiceConfig
from app.voice.tts.factory import create_tts_provider

async def test_tts():
    cfg = VoiceConfig.from_settings()
    print("Resolved config tts_provider:", cfg.tts_provider)
    print("Resolved config voice_id:", cfg.elevenlabs_voice_id)
    print("Resolved config model_id:", cfg.elevenlabs_model_id)
    tts = create_tts_provider(cfg)
    print("TTS Provider resolved:", type(tts).__name__)
    print("Primary provider name:", getattr(tts, "provider_name", "unknown"))
    t0 = time.perf_counter()
    try:
        res = await tts.synthesize(text="Hello! This is a test from ElevenLabs.")
        ms = (time.perf_counter() - t0) * 1000
        print(f"TTS Synthesized successfully in {ms:.1f}ms!")
        print(f"Audio bytes length: {len(res.audio_bytes)}")
        print(f"Format: {res.content_type}")
        print(f"Duration estimate: {res.duration_seconds:.2f}s")
        print(f"Provider used: {getattr(tts, 'last_provider_used', getattr(tts, 'provider_name', 'unknown'))}")
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"TTS FAILED in {ms:.1f}ms: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
