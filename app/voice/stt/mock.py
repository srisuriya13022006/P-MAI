"""
P12 — Deterministic Mock Speech-to-Text Provider.
Used for deterministic testing and local development without cloud dependencies.
"""
import asyncio
from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.base import SpeechToTextProvider


class MockSTTProvider(SpeechToTextProvider):
    """
    Deterministic mock STT provider for unit and regression tests.
    Does not call external APIs.
    """

    def __init__(
        self,
        default_transcript: str = "What time is it in Tokyo?",
        simulate_timeout: bool = False,
        simulate_unavailable: bool = False,
        simulate_empty: bool = False,
        simulated_latency_seconds: float = 0.0,
        confidence: float = 1.0,
    ):
        self.default_transcript = default_transcript
        self.simulate_timeout = simulate_timeout
        self.simulate_unavailable = simulate_unavailable
        self.simulate_empty = simulate_empty
        self.simulated_latency_seconds = simulated_latency_seconds
        self.confidence = confidence
        self._transcript_queue: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def enqueue_transcript(self, transcript: str) -> None:
        """Enqueue a transcript to be returned on subsequent call."""
        self._transcript_queue.append(transcript)

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        if self.simulated_latency_seconds > 0:
            await asyncio.sleep(self.simulated_latency_seconds)

        if self.simulate_unavailable:
            raise STTUnavailableError("Mock STT provider is intentionally unavailable.")

        if self.simulate_timeout:
            raise STTTimeoutError("Mock STT transcription timed out.")

        if self.simulate_empty:
            raise EmptyTranscriptionError("Mock STT produced an empty transcription.")

        if self._transcript_queue:
            transcript = self._transcript_queue.pop(0)
        else:
            transcript = self.default_transcript

        if not transcript or not transcript.strip():
            raise EmptyTranscriptionError("Speech-to-Text produced empty or whitespace transcript.")

        # Estimate duration roughly from bytes
        duration = len(audio_bytes) / 32000.0 if audio_bytes else 1.0

        return STTResult(
            transcript=transcript.strip(),
            language=language or "en",
            duration_seconds=round(duration, 2),
            provider=self.provider_name,
            confidence=self.confidence,
        )
