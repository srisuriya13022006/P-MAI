import asyncio
import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv("d:/suriya\projects\P-MAI\.env")

from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
from app.voice.tts.preprocessor import SpeechTextPreprocessor

async def main():
    print("Testing live ElevenLabs streaming with segmentation...")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: No ELEVENLABS_API_KEY")
        return

    adapter = ElevenLabsTTSAdapter(
        api_key=api_key,
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_v4_turbo"),
        max_segment_chars=35,  # Force 2 segments with minimal characters
    )

    test_text = "Good evening Arthur. The beacon is lit tonight."
    preprocessor = SpeechTextPreprocessor()
    prep = preprocessor.preprocess(test_text, target_provider="elevenlabs", max_segment_chars=35)
    print(f"Prepared segments ({len(prep.segments)}): {prep.segments}")

    chunk_count = 0
    total_bytes = 0
    async for chunk in adapter.synthesize_stream(test_text):
        chunk_count += 1
        total_bytes += len(chunk.audio_bytes)

    print(f"Live test complete!")
    print(f"Total chunks: {chunk_count}")
    print(f"Total audio bytes: {total_bytes}")
    approx_duration_s = total_bytes / (16000 * 2)
    print(f"Approx duration: {approx_duration_s:.2f} seconds")

    if chunk_count > 0 and total_bytes > 0:
        print("LIVE_VERIFICATION_SUCCESS")
    else:
        print("LIVE_VERIFICATION_FAILED")

if __name__ == "__main__":
    asyncio.run(main())
