"""
P13 — Production Whisper Speech-to-Text Adapter.
Provides concrete HTTP/REST and streaming adapter for OpenAI/Groq Whisper STT API.
Implements SpeechToTextProvider and StreamingSpeechToTextProvider.
"""
import io
import time
from typing import Any
import httpx

from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.streaming import StreamingSpeechToTextProvider


class WhisperSTTAdapter(StreamingSpeechToTextProvider):
    """
    Production Whisper STT Adapter (OpenAI / Groq Whisper API compatible).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "whisper-large-v3-turbo",
        timeout_seconds: float = 15.0,
        max_reconnect_attempts: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_reconnect_attempts = max_reconnect_attempts

        self._audio_buffer = bytearray()
        self._last_partial_transcript: str = ""

    @property
    def provider_name(self) -> str:
        return "whisper"

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """Execute full transcription against Whisper endpoint."""
        if not self.api_key:
            raise STTUnavailableError("STT API key is not configured.")

        if not audio_bytes or len(audio_bytes) == 0:
            raise EmptyTranscriptionError("Audio buffer is empty.")

        url = f"{self.base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Format multipart file upload
        files = {
            "file": ("audio.wav", audio_bytes, content_type or "audio/wav"),
        }
        data = {
            "model": self.model,
            "response_format": "json",
        }
        if language:
            data["language"] = language

        attempts = 0
        while attempts <= self.max_reconnect_attempts:
            attempts += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, files=files, data=data)

                if resp.status_code == 401 or resp.status_code == 403:
                    raise STTUnavailableError("STT authentication failed (invalid API key).")
                elif resp.status_code >= 500:
                    if attempts <= self.max_reconnect_attempts:
                        continue
                    raise STTUnavailableError(f"STT upstream server error: HTTP {resp.status_code}.")
                elif resp.status_code != 200:
                    raise STTUnavailableError(f"STT transcription failed: HTTP {resp.status_code}.")

                payload = resp.json()
                text = (payload.get("text") or "").strip()
                if not text:
                    raise EmptyTranscriptionError("Whisper returned empty transcript.")

                return STTResult(
                    transcript=text,
                    language=language or "en",
                    provider=self.provider_name,
                    confidence=1.0,
                )

            except httpx.TimeoutException:
                if attempts <= self.max_reconnect_attempts:
                    continue
                raise STTTimeoutError("STT provider request timed out.")
            except httpx.RequestError as req_err:
                if attempts <= self.max_reconnect_attempts:
                    continue
                raise STTUnavailableError(f"STT provider network error: {type(req_err).__name__}")

        raise STTUnavailableError("STT provider unavailable after reconnect attempts.")

    async def feed_audio_chunk(
        self,
        chunk: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        """
        Buffer streaming chunks.
        Optionally generates low-latency partial updates when buffer reaches threshold.
        """
        if not self.api_key:
            raise STTUnavailableError("STT API key is not configured.")

        self._audio_buffer.extend(chunk)
        # Note: partial transcript generation on live whisper is gated to avoid excessive API spam
        return []

    async def finalize_stream(self) -> STTResult:
        """Finalize accumulated stream buffer."""
        audio_data = bytes(self._audio_buffer)
        self.reset()
        return await self.transcribe(audio_data)

    def reset(self) -> None:
        self._audio_buffer.clear()
        self._last_partial_transcript = ""
