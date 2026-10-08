"""
P12 — Deterministic Mock Text-to-Speech Provider.
Generates valid WAV audio bytes deterministically without cloud dependencies.
"""
import asyncio
import io
import wave
from app.voice.exceptions import TTSTimeoutError, TTSUnavailableError
from app.voice.schemas import TTSResult
from app.voice.tts.base import TextToSpeechProvider


import math
import struct


def generate_mock_wav(duration_seconds: float = 1.0, framerate: int = 16000, frequency_hz: float = 440.0) -> bytes:
    """Generate minimal valid PCM WAV bytes for testing with detectable acoustic energy."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        n_samples = max(1, int(duration_seconds * framerate))
        samples = bytearray()
        for i in range(n_samples):
            val = int(10000.0 * math.sin(2.0 * math.pi * frequency_hz * i / framerate))
            samples.extend(struct.pack("<h", val))
        wf.writeframes(samples)
    return buf.getvalue()


class MockTTSProvider(TextToSpeechProvider):
    """
    Deterministic mock TTS provider for testing.
    """

    def __init__(
        self,
        simulate_timeout: bool = False,
        simulate_unavailable: bool = False,
        simulated_latency_seconds: float = 0.0,
        audio_content_type: str = "audio/wav",
    ):
        self.simulate_timeout = simulate_timeout
        self.simulate_unavailable = simulate_unavailable
        self.simulated_latency_seconds = simulated_latency_seconds
        self.audio_content_type = audio_content_type
        self.last_spoken_text: str | None = None

    @property
    def provider_name(self) -> str:
        return "mock"

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        if self.simulated_latency_seconds > 0:
            await asyncio.sleep(self.simulated_latency_seconds)

        if self.simulate_unavailable:
            raise TTSUnavailableError("Mock TTS provider is intentionally unavailable.")

        if self.simulate_timeout:
            raise TTSTimeoutError("Mock TTS synthesis timed out.")

        self.last_spoken_text = text

        # Estimate duration based on word count: ~150 words per minute -> 2.5 words/sec
        words = len(text.split()) if text else 1
        est_duration = max(0.5, round(words / 2.5, 2))
        audio_bytes = generate_mock_wav(duration_seconds=est_duration)

        return TTSResult(
            audio_bytes=audio_bytes,
            content_type=self.audio_content_type,
            duration_seconds=est_duration,
            provider=self.provider_name,
        )
