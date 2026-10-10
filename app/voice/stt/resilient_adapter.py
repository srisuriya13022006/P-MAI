"""
P15.4 — Resilient Speech-to-Text Adapter with Local Whisper Fallback.
Guarantees uninterrupted speech recognition by routing to AssemblyAI first,
falling back gracefully to Local Whisper on network/cloud unavailability without flapping loops.
"""
import logging
from typing import Any

from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.streaming import StreamingSpeechToTextProvider

logger = logging.getLogger(__name__)


class ResilientSTTAdapter(StreamingSpeechToTextProvider):
    """
    Manages primary AssemblyAI streaming STT with controlled Local Whisper fallback.
    Prevents provider flapping and logs structured, secret-free transition events.
    """

    def __init__(
        self,
        primary_provider: SpeechToTextProvider,
        fallback_provider: SpeechToTextProvider,
        fallback_enabled: bool = True,
    ):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.fallback_enabled = fallback_enabled

        self.stt_mode: str = "CLOUD"
        self.last_provider_used: str = primary_provider.provider_name
        self._buffered_audio = bytearray()

    @property
    def provider_name(self) -> str:
        return self.last_provider_used

    async def feed_audio_chunk(
        self,
        chunk: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        """
        Buffer chunk and pass to primary streaming provider if in CLOUD mode.
        """
        self._buffered_audio.extend(chunk)

        if self.stt_mode == "CLOUD" and isinstance(self.primary_provider, StreamingSpeechToTextProvider):
            try:
                return await self.primary_provider.feed_audio_chunk(chunk, content_type)
            except Exception as exc:
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                cat = classify_cloud_error(exc)
                if cat == CloudErrorCategory.HARD_CONFIG:
                    self.stt_mode = "OFFLINE"
                    logger.error(
                        "Primary STT %s encountered hard configuration/auth failure (%s). Fallback inhibited.",
                        getattr(self.primary_provider, "provider_name", "assemblyai"),
                        type(exc).__name__,
                    )
                    raise exc

                if not self.fallback_enabled or self.fallback_provider is None:
                    self.stt_mode = "OFFLINE"
                    raise

                # Switch to local fallback for this utterance
                logger.exception("Primary STT failed in feed_audio_chunk: %s", exc)
                fallback_reason = "rate_limit_429" if cat == CloudErrorCategory.RATE_LIMIT else type(exc).__name__
                logger.warning(
                    "STT provider: %s\nSTT fallback reason: %s\nSTT provider switched to: %s",
                    getattr(self.primary_provider, "provider_name", "assemblyai"),
                    fallback_reason,
                    getattr(self.fallback_provider, "provider_name", "local"),
                )
                self.stt_mode = "LOCAL_FALLBACK"

        return []

    async def finalize_stream(self) -> STTResult:
        """
        Finalize stream via primary, gracefully falling back to Local Whisper if primary fails.
        """
        if self.stt_mode == "CLOUD":
            try:
                if isinstance(self.primary_provider, StreamingSpeechToTextProvider):
                    res = await self.primary_provider.finalize_stream()
                else:
                    res = await self.primary_provider.transcribe(bytes(self._buffered_audio))
                self.last_provider_used = self.primary_provider.provider_name
                self.reset()
                return res
            except Exception as exc:
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                cat = classify_cloud_error(exc)
                if cat == CloudErrorCategory.HARD_CONFIG:
                    self.stt_mode = "OFFLINE"
                    logger.error(
                        "Primary STT %s encountered hard configuration/auth failure (%s). Fallback inhibited.",
                        getattr(self.primary_provider, "provider_name", "assemblyai"),
                        type(exc).__name__,
                    )
                    raise exc

                if not self.fallback_enabled or self.fallback_provider is None:
                    self.stt_mode = "OFFLINE"
                    raise

                logger.exception("Primary STT failed in finalize_stream: %s", exc)
                fallback_reason = "rate_limit_429" if cat == CloudErrorCategory.RATE_LIMIT else type(exc).__name__
                logger.warning(
                    "STT provider: %s\nSTT fallback reason: %s\nSTT provider switched to: %s",
                    getattr(self.primary_provider, "provider_name", "assemblyai"),
                    fallback_reason,
                    getattr(self.fallback_provider, "provider_name", "local"),
                )
                self.stt_mode = "LOCAL_FALLBACK"

        # Local Fallback execution
        if not self._buffered_audio:
            raise EmptyTranscriptionError("Audio buffer is empty for fallback STT.")

        audio_bytes = bytes(self._buffered_audio)
        self.reset()
        res = await self.fallback_provider.transcribe(audio_bytes)
        self.last_provider_used = self.fallback_provider.provider_name
        return res

    def reset(self) -> None:
        """Reset internal buffers and providers."""
        self._buffered_audio.clear()
        if hasattr(self.primary_provider, "reset"):
            self.primary_provider.reset()
        if hasattr(self.fallback_provider, "reset"):
            self.fallback_provider.reset()

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """
        Batch transcription: attempts primary first, falls back to Local Whisper on failure.
        """
        if self.stt_mode == "CLOUD":
            try:
                res = await self.primary_provider.transcribe(audio_bytes, content_type, language)
                self.last_provider_used = self.primary_provider.provider_name
                return res
            except Exception as exc:
                from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
                cat = classify_cloud_error(exc)
                if cat == CloudErrorCategory.HARD_CONFIG:
                    self.stt_mode = "OFFLINE"
                    logger.error(
                        "Primary STT %s encountered hard configuration/auth failure (%s). Fallback inhibited.",
                        getattr(self.primary_provider, "provider_name", "assemblyai"),
                        type(exc).__name__,
                    )
                    raise exc

                if not self.fallback_enabled or self.fallback_provider is None:
                    self.stt_mode = "OFFLINE"
                    raise

                fallback_reason = "rate_limit_429" if cat == CloudErrorCategory.RATE_LIMIT else type(exc).__name__
                logger.warning(
                    "STT provider: %s\nSTT fallback reason: %s\nSTT provider switched to: %s",
                    getattr(self.primary_provider, "provider_name", "assemblyai"),
                    fallback_reason,
                    getattr(self.fallback_provider, "provider_name", "local"),
                )
                self.stt_mode = "LOCAL_FALLBACK"

        # Fallback
        res = await self.fallback_provider.transcribe(audio_bytes, content_type, language)
        self.last_provider_used = self.fallback_provider.provider_name
        return res

