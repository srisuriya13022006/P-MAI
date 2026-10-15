import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

async def test_long_text_limits():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("No ELEVENLABS_API_KEY found.")
        return

    from app.voice.tts.elevenlabs_adapter import ElevenLabsTTSAdapter
    from app.voice.tts.preprocessor import SpeechTextPreprocessor

    adapter = ElevenLabsTTSAdapter(
        api_key=api_key,
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_v4_turbo"),
    )

    # Let's check a 6-paragraph story
    paragraphs = [
        "Once upon a time, in a small harbor town nestled between towering cliffs, an old lighthouse keeper named Arthur watched the evening tide roll in.",
        "Every twilight, he climbed the spiral staircase of worn granite, carrying a brass lantern and a heart full of memories.",
        "The sea had taken much from him over the years, yet it remained his truest companion, whispering forgotten songs against the rocks.",
        "One stormy night, when the fog grew so dense that even the seagulls fell silent, a faint golden light flickered in the distance.",
        "Arthur adjusted his spectacles and realized that a small wooden vessel was navigating dangerously close to the sunken reef.",
        "With steady hands, he turned the great crystal prism, casting a brilliant beam across the water and guiding the travelers safely home to the harbor."
    ]
    full_story = "\n\n".join(paragraphs)
    print(f"Full story character count: {len(full_story)}, word count: {len(full_story.split())}")

    prep = SpeechTextPreprocessor()
    res = prep.preprocess(full_story, target_provider="elevenlabs", model_id="eleven_v4_turbo")
    print(f"Preprocessed text length: {len(res.expressive_text)}")
    print(f"Audio tag: {res.audio_tag}")

if __name__ == "__main__":
    asyncio.run(test_long_text_limits())
