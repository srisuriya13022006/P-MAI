"""
P15.5 — Resilient Text-to-Speech Adapter with Local Fallback.
Guarantees uninterrupted voice synthesis by routing to ElevenLabs first,
falling back gracefully to Local pyttsx3 on network/service failure without flapping loops.
Strictly surfaces hard configuration/authentication errors.
"""
import logging
from typing import Any, AsyncGenerator, Optional

from app.core.runtime_mode import CloudErrorCategory, classify_cloud_error
from app.voice.exceptions import (
    TTSError,
    TTSTimeoutError,
    TTSUnavailableError,
)
from app.voice.schemas import TTSResult
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.preprocessor import SpeechTextPreprocessor
from app.voice.tts.streaming import StreamingTextToSpeechProvider

logger = logging.getLogger(__name__)


def _format_diagnostics(exc: Exception) -> str:
    """Format safe diagnostic metadata without secrets."""
    parts = []
    orig_cls = getattr(exc, "original_exception_class", type(exc).__name__)
    if orig_cls:
        parts.append(f"exception={orig_cls}")
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"http_status={status_code}")
    provider_code = getattr(exc, "provider_error_code", None)
    if provider_code:
        parts.append(f"provider_code={provider_code}")
    model_id = getattr(exc, "model_id", None)
    if model_id:
        parts.append(f"model={model_id}")
    voice_id = getattr(exc, "voice_id", None)
    if voice_id:
        parts.append(f"voice={voice_id}")
    operation = getattr(exc, "api_operation", None)
    if operation:
        parts.append(f"operation={operation}")
    endpoint = getattr(exc, "endpoint_path", None)
    if endpoint:
        parts.append(f"endpoint={endpoint}")
    phase = getattr(exc, "failure_phase", None)
    if phase:
        parts.append(f"phase={phase}")
    sanitized_msg = getattr(exc, "sanitized_message", None) or str(exc)
    if sanitized_msg:
        clean_msg = sanitized_msg.replace("\n", " ").strip()
        parts.append(f"message='{clean_msg}'")
    return ", ".join(parts)


def _resolve_fallback_reason(exc: Exception, cat: CloudErrorCategory) -> str:
    """Resolve specific, actionable fallback reason without generic masking."""
    if cat == CloudErrorCategory.RATE_LIMIT:
        return "rate_limit_429"
    provider_code = getattr(exc, "provider_error_code", None)
    status_code = getattr(exc, "status_code", None)
    if provider_code and status_code:
        return f"{provider_code} (HTTP {status_code})"
    if provider_code:
        return str(provider_code)
    if status_code:
        return f"HTTP_{status_code}"
    orig_cls = getattr(exc, "original_exception_class", None)
    if orig_cls and orig_cls != type(exc).__name__:
        return f"{type(exc).__name__}[{orig_cls}]"
    return type(exc).__name__


class ResilientTTSAdapter(StreamingTextToSpeechProvider):
    """
    Manages primary ElevenLabs streaming TTS with controlled Local pyttsx3 fallback.
    Prevents provider flapping and logs structured, secret-free transition events.
    """

    def __init__(
        self,
        primary_provider: TextToSpeechProvider,
        fallback_provider: Optional[TextToSpeechProvider] = None,
        fallback_enabled: bool = True,
        preprocessor: Optional[SpeechTextPreprocessor] = None,
    ):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.fallback_enabled = fallback_enabled
        self.preprocessor = preprocessor or SpeechTextPreprocessor()

        self.tts_mode: str = "CLOUD"
        self.last_provider_used: str = primary_provider.provider_name

    @property
    def provider_name(self) -> str:
        return self.last_provider_used

    def reset_mode(self) -> None:
        """Reset mode back to CLOUD for subsequent sessions or test assertions."""
        self.tts_mode = "CLOUD"
        self.last_provider_used = self.primary_provider.provider_name

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize text via primary ElevenLabs, falling back to Local pyttsx3 if primary fails.
        """
        if self.tts_mode == "CLOUD":
            try:
                res = await self.primary_provider.synthesize(text, voice=voice, language=language)
                self.last_provider_used = self.primary_provider.provider_name
                return res
            except Exception as exc:
                cat = classify_cloud_error(exc)
                diag_info = _format_diagnostics(exc)
                fallback_reason = _resolve_fallback_reason(exc, cat)

                if cat == CloudErrorCategory.HARD_CONFIG:
                    self.tts_mode = "OFFLINE"
                    logger.error(
                        "Primary TTS %s encountered authentication/configuration failure (%s). Diagnostics: %s",
                        getattr(self.primary_provider, "provider_name", "elevenlabs"),
                        type(exc).__name__,
                        diag_info,
                    )
                    if not self.fallback_enabled or self.fallback_provider is None:
                        raise exc
                    logger.warning(
                        "Falling back to local TTS provider '%s' to preserve speech output without remote retries.",
                        getattr(self.fallback_provider, "provider_name", "local"),
                    )
                else:
                    if not self.fallback_enabled or self.fallback_provider is None:
                        self.tts_mode = "OFFLINE"
                        raise exc

                    # Switch to local fallback
                    logger.warning(
                        "TTS provider: %s\nTTS fallback reason: %s\nTTS provider switched to: %s\nTTS diagnostics: %s",
                        getattr(self.primary_provider, "provider_name", "elevenlabs"),
                        fallback_reason,
                        getattr(self.fallback_provider, "provider_name", "local"),
                        diag_info,
                    )
                    self.tts_mode = "LOCAL_FALLBACK"

        # Fallback execution: strip audio tags so pyttsx3 does not vocalize them
        clean_text = self.preprocessor.strip_audio_tags(text)
        try:
            res = await self.fallback_provider.synthesize(clean_text, voice=voice, language=language)
            self.last_provider_used = self.fallback_provider.provider_name
            logger.info(
                "Local TTS fallback succeeded via '%s' (audio_bytes=%d, duration=%.2fs).",
                getattr(self.fallback_provider, "provider_name", "local"),
                len(res.audio_bytes),
                res.duration_seconds,
            )
            return res
        except Exception as fb_exc:
            logger.error(
                "Local TTS fallback failed via '%s': %s",
                getattr(self.fallback_provider, "provider_name", "local"),
                fb_exc,
            )
            raise fb_exc

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Stream synthesized chunks via primary ElevenLabs, falling back to Local pyttsx3 if primary fails.
        Handles long-form segmented synthesis safely:
        - If failure occurs after streaming started, only remaining un-emitted segments are synthesized
          to prevent duplicated speech.
        - Records whether error occurred before or after audio was emitted.
        """
        prep = self.preprocessor.preprocess(text)
        segments = prep.segments if prep.segments else [text]

        chunks_emitted = 0
        failed_seg_idx = 0

        if self.tts_mode == "CLOUD":
            try:
                for s_idx, seg in enumerate(segments):
                    failed_seg_idx = s_idx
                    if isinstance(self.primary_provider, StreamingTextToSpeechProvider):
                        async for chunk in self.primary_provider.synthesize_stream(
                            text=seg,
                            voice=voice,
                            language=language,
                            chunk_duration_seconds=chunk_duration_seconds,
                        ):
                            self.last_provider_used = self.primary_provider.provider_name
                            chunks_emitted += 1
                            yield chunk
                    else:
                        res = await self.primary_provider.synthesize(seg, voice=voice, language=language)
                        self.last_provider_used = self.primary_provider.provider_name
                        chunks_emitted += 1
                        yield res
                return
            except Exception as exc:
                cat = classify_cloud_error(exc)
                diag_info = _format_diagnostics(exc)
                fallback_reason = _resolve_fallback_reason(exc, cat)
                audio_emitted_before_error = chunks_emitted > 0

                logger.warning(
                    "Primary TTS failed on segment %d/%d (audio_emitted_before_error=%s, chunks_emitted=%d).\n"
                    "TTS fallback reason: %s\nTTS diagnostics: %s",
                    failed_seg_idx + 1,
                    len(segments),
                    audio_emitted_before_error,
                    chunks_emitted,
                    fallback_reason,
                    diag_info,
                )

                if cat == CloudErrorCategory.HARD_CONFIG and not audio_emitted_before_error:
                    self.tts_mode = "OFFLINE"
                    logger.error(
                        "Primary TTS %s encountered authentication/configuration failure (%s). Diagnostics: %s",
                        getattr(self.primary_provider, "provider_name", "elevenlabs"),
                        type(exc).__name__,
                        diag_info,
                    )
                    if not self.fallback_enabled or self.fallback_provider is None:
                        raise exc
                    logger.warning(
                        "Falling back to local TTS provider '%s' to preserve speech output without remote retries.",
                        getattr(self.fallback_provider, "provider_name", "local"),
                    )
                else:
                    if not self.fallback_enabled or self.fallback_provider is None:
                        if not audio_emitted_before_error:
                            self.tts_mode = "OFFLINE"
                        raise exc

                    # Switch to local fallback
                    logger.warning(
                        "TTS provider switched to: %s for %d remaining segments.",
                        getattr(self.fallback_provider, "provider_name", "local"),
                        len(segments) - failed_seg_idx,
                    )
                    self.tts_mode = "LOCAL_FALLBACK"

        # Fallback execution: only synthesize remaining un-emitted segments to prevent duplicated speech
        remaining_segments = segments[failed_seg_idx:] if self.tts_mode == "LOCAL_FALLBACK" and chunks_emitted > 0 else segments
        fb_chunk_count = 0
        fb_total_bytes = 0

        for rem_seg in remaining_segments:
            clean_text = self.preprocessor.strip_audio_tags(rem_seg)
            try:
                if isinstance(self.fallback_provider, StreamingTextToSpeechProvider):
                    async for chunk in self.fallback_provider.synthesize_stream(
                        text=clean_text,
                        voice=voice,
                        language=language,
                        chunk_duration_seconds=chunk_duration_seconds,
                    ):
                        self.last_provider_used = self.fallback_provider.provider_name
                        fb_chunk_count += 1
                        fb_total_bytes += len(chunk.audio_bytes)
                        yield chunk
                else:
                    res = await self.fallback_provider.synthesize(clean_text, voice=voice, language=language)
                    self.last_provider_used = self.fallback_provider.provider_name
                    fb_chunk_count += 1
                    fb_total_bytes += len(res.audio_bytes)
                    yield res
            except Exception as fb_exc:
                logger.error(
                    "Local TTS fallback streaming failed via '%s': %s",
                    getattr(self.fallback_provider, "provider_name", "local"),
                    fb_exc,
                )
                raise fb_exc

        logger.info(
            "Local TTS fallback streaming succeeded via '%s' (chunks=%d, total_bytes=%d).",
            getattr(self.fallback_provider, "provider_name", "local"),
            fb_chunk_count,
            fb_total_bytes,
        )
