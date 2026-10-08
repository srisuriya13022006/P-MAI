"""
P12.2 — Streaming Speech-to-Text Provider Abstraction.
Supports incremental chunk ingestion, partial transcripts, and finalization.
"""
from abc import ABC, abstractmethod
import asyncio
from typing import AsyncGenerator

from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.base import SpeechToTextProvider


class StreamingSpeechToTextProvider(SpeechToTextProvider, ABC):
    """
    Abstract interface for streaming / low-latency STT providers.
    """

    @abstractmethod
    async def feed_audio_chunk(
        self,
        chunk: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        """
        Process an incremental audio chunk.
        May yield zero or more partial STTResult items.
        """
        pass

    @abstractmethod
    async def finalize_stream(self) -> STTResult:
        """
        Signal end of current speech utterance and return the final STTResult.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal streaming state for a new utterance."""
        pass


class MockStreamingSTTProvider(StreamingSpeechToTextProvider):
    """
    Deterministic mock streaming STT provider for real-time testing.
    """

    def __init__(
        self,
        final_transcript: str = "What time is it in Tokyo?",
        partial_transcripts: list[str] | None = None,
        simulate_timeout: bool = False,
        simulate_unavailable: bool = False,
        simulate_disconnect: bool = False,
        simulate_empty: bool = False,
    ):
        self.final_transcript = final_transcript
        self.partial_transcripts = partial_transcripts or []
        self.simulate_timeout = simulate_timeout
        self.simulate_unavailable = simulate_unavailable
        self.simulate_disconnect = simulate_disconnect
        self.simulate_empty = simulate_empty
        self._partial_index = 0
        self._audio_chunks_received = 0
        self._transcript_queue: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock_streaming"

    def enqueue_transcript(self, transcript: str) -> None:
        """Enqueue next final transcript for multi-turn testing."""
        self._transcript_queue.append(transcript)

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """Fallback non-streaming transcribe method."""
        return await self.finalize_stream()

    async def feed_audio_chunk(
        self,
        chunk: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        if self.simulate_unavailable:
            raise STTUnavailableError("Mock streaming STT is unavailable.")
        if self.simulate_disconnect:
            raise STTUnavailableError("Mock streaming STT disconnected unexpectedly.")
        if self.simulate_timeout:
            raise STTTimeoutError("Mock streaming STT timed out.")

        self._audio_chunks_received += 1
        results: list[STTResult] = []

        # If partial transcripts configured, yield next one
        if self.partial_transcripts and self._partial_index < len(self.partial_transcripts):
            partial_text = self.partial_transcripts[self._partial_index]
            self._partial_index += 1
            results.append(
                STTResult(
                    transcript=partial_text,
                    confidence=0.85,
                    provider=self.provider_name,
                )
            )
        elif not self.partial_transcripts and self.final_transcript:
            # Emit first word or prefix as a partial transcript on first chunk
            words = self.final_transcript.split()
            if words and self._audio_chunks_received == 1:
                results.append(
                    STTResult(
                        transcript=" ".join(words[: max(1, len(words) // 2)]),
                        confidence=0.8,
                        provider=self.provider_name,
                    )
                )

        return results

    async def finalize_stream(self) -> STTResult:
        if self.simulate_unavailable or self.simulate_disconnect:
            raise STTUnavailableError("Mock streaming STT provider is unavailable.")
        if self.simulate_timeout:
            raise STTTimeoutError("Mock streaming STT timed out during finalization.")
        if self.simulate_empty:
            raise EmptyTranscriptionError("Mock streaming STT produced empty transcript.")

        if self._transcript_queue:
            target = self._transcript_queue.pop(0)
        else:
            target = self.final_transcript

        if not target or not target.strip():
            raise EmptyTranscriptionError("Final transcript is empty.")

        return STTResult(
            transcript=target.strip(),
            confidence=1.0,
            provider=self.provider_name,
        )

    def reset(self) -> None:
        self._partial_index = 0
        self._audio_chunks_received = 0
