"""
P12.2 — Streaming Text-to-Speech Provider Abstraction.
Yields incremental audio chunks for low-latency playback of final grounded answers.
"""
from abc import ABC, abstractmethod
import asyncio
from typing import AsyncGenerator

from app.voice.exceptions import (
    TTSTimeoutError,
    TTSUnavailableError,
)
from app.voice.schemas import TTSResult
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.mock import generate_mock_wav


class StreamingTextToSpeechProvider(TextToSpeechProvider, ABC):
    """
    Abstract interface for streaming / chunked TTS providers.
    """

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Yield incremental synthesized audio chunks for the grounded text response.
        """
        pass


class MockStreamingTTSProvider(StreamingTextToSpeechProvider):
    """
    Deterministic mock streaming TTS provider.
    Yields small valid WAV chunks representing spoken audio.
    """

    def __init__(
        self,
        simulate_timeout: bool = False,
        simulate_unavailable: bool = False,
        simulate_disconnect: bool = False,
        chunk_count: int = 3,
    ):
        self.simulate_timeout = simulate_timeout
        self.simulate_unavailable = simulate_unavailable
        self.simulate_disconnect = simulate_disconnect
        self.chunk_count = chunk_count
        self.last_spoken_text: str | None = None

    @property
    def provider_name(self) -> str:
        return "mock_streaming"

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        self.last_spoken_text = text
        if self.simulate_unavailable:
            raise TTSUnavailableError("Mock streaming TTS is unavailable.")
        if self.simulate_timeout:
            raise TTSTimeoutError("Mock streaming TTS timed out.")
        audio_bytes = generate_mock_wav(duration_seconds=1.0)
        return TTSResult(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            duration_seconds=1.0,
            provider=self.provider_name,
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        self.last_spoken_text = text
        if self.simulate_unavailable:
            raise TTSUnavailableError("Mock streaming TTS is unavailable.")
        if self.simulate_timeout:
            raise TTSTimeoutError("Mock streaming TTS timed out.")

        words = text.split() if text else ["OK"]
        step = max(1, len(words) // self.chunk_count)

        for i in range(self.chunk_count):
            if self.simulate_disconnect and i == 1:
                raise TTSUnavailableError("Mock streaming TTS disconnected during chunk generation.")

            sub_words = words[i * step : (i + 1) * step]
            chunk_audio = generate_mock_wav(duration_seconds=chunk_duration_seconds)
            yield TTSResult(
                audio_bytes=chunk_audio,
                content_type="audio/wav",
                duration_seconds=chunk_duration_seconds,
                provider=self.provider_name,
            )
            # Minimal simulated delay between chunks
            await asyncio.sleep(0.01)
