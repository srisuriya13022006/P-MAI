"""
P13 — Production Text-to-Speech Adapter.
Provides concrete HTTP streaming adapter for OpenAI-compatible audio speech synthesis.
Implements TextToSpeechProvider and StreamingTextToSpeechProvider.
"""
from typing import AsyncGenerator
import httpx

from app.voice.exceptions import (
    TTSError,
    TTSTimeoutError,
    TTSUnavailableError,
)
from app.voice.schemas import TTSResult
from app.voice.tts.streaming import StreamingTextToSpeechProvider


class OpenAITTSAdapter(StreamingTextToSpeechProvider):
    """
    Production TTS Adapter (OpenAI / compatible Audio Speech API).
    Supports streaming response chunks, configurable voices, formats, and timeouts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        default_voice: str = "alloy",
        audio_format: str = "wav",
        timeout_seconds: float = 15.0,
        max_reconnect_attempts: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_voice = default_voice
        self.audio_format = audio_format
        self.timeout_seconds = timeout_seconds
        self.max_reconnect_attempts = max_reconnect_attempts

        self.last_spoken_text: str | None = None

    @property
    def provider_name(self) -> str:
        return "openai_tts"

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        """Synthesize complete audio response."""
        if not self.api_key:
            raise TTSUnavailableError("TTS API key is not configured.")

        if not text or not text.strip():
            return TTSResult(audio_bytes=b"", content_type=f"audio/{self.audio_format}", provider=self.provider_name)

        self.last_spoken_text = text
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": text,
            "voice": voice or self.default_voice,
            "response_format": self.audio_format,
        }

        attempts = 0
        while attempts <= self.max_reconnect_attempts:
            attempts += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=data)

                if resp.status_code == 401 or resp.status_code == 403:
                    raise TTSUnavailableError("TTS authentication failed (invalid API key).")
                elif resp.status_code >= 500:
                    if attempts <= self.max_reconnect_attempts:
                        continue
                    raise TTSUnavailableError(f"TTS upstream server error: HTTP {resp.status_code}.")
                elif resp.status_code != 200:
                    raise TTSUnavailableError(f"TTS synthesis failed: HTTP {resp.status_code}.")

                audio_data = resp.content
                return TTSResult(
                    audio_bytes=audio_data,
                    content_type=f"audio/{self.audio_format}",
                    provider=self.provider_name,
                )

            except httpx.TimeoutException:
                if attempts <= self.max_reconnect_attempts:
                    continue
                raise TTSTimeoutError("TTS synthesis timed out.")
            except httpx.RequestError as req_err:
                if attempts <= self.max_reconnect_attempts:
                    continue
                raise TTSUnavailableError(f"TTS provider network error: {type(req_err).__name__}")

        raise TTSUnavailableError("TTS provider unavailable after reconnect attempts.")

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        """Stream synthesized audio chunks for low-latency playback."""
        if not self.api_key:
            raise TTSUnavailableError("TTS API key is not configured.")

        self.last_spoken_text = text
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": text,
            "voice": voice or self.default_voice,
            "response_format": self.audio_format,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream("POST", url, headers=headers, json=data) as resp:
                    if resp.status_code != 200:
                        raise TTSUnavailableError(f"Streaming TTS failed: HTTP {resp.status_code}.")

                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield TTSResult(
                                audio_bytes=chunk,
                                content_type=f"audio/{self.audio_format}",
                                provider=self.provider_name,
                            )
        except httpx.TimeoutException:
            raise TTSTimeoutError("Streaming TTS request timed out.")
        except httpx.RequestError as err:
            raise TTSUnavailableError(f"Streaming TTS network error: {err}")
