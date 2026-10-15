"""
P15.5 — Production ElevenLabs Text-to-Speech Adapter.
Integrates ElevenLabs streaming speech synthesis, canonical 16kHz PCM audio delivery,
configurable voice settings, and strict authentication/error classifications.
Supports both Text-to-Dialogue (for eleven_v4_turbo) and Text-to-Speech endpoints.
"""
import asyncio
import inspect
import io
import logging
import time
from typing import AsyncGenerator, Optional
import httpx

from app.voice.audio import CanonicalAudioFormat, pcm_to_wav
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

# Default free-tier compatible premade fallback voice
PREMADE_FALLBACK_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Bella (premade account voice)


class ElevenLabsTTSAdapter(StreamingTextToSpeechProvider):
    """
    Primary Cloud Text-to-Speech Provider using ElevenLabs.
    Delivers canonical 16kHz 16-bit Mono PCM chunks natively without transcoding.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "EXAVITQu4vr4xnSDxMaL",  # Default standard clear premade voice
        model_id: str = "eleven_v4_turbo",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        optimize_streaming_latency: int = 3,
        timeout_seconds: float = 15.0,
        output_format: str = "pcm_16000",
        max_segment_chars: int = 1500,
        preprocessor: Optional[SpeechTextPreprocessor] = None,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.optimize_streaming_latency = optimize_streaming_latency
        self.timeout_seconds = timeout_seconds
        self.output_format = output_format
        self.max_segment_chars = max_segment_chars

        self.preprocessor = preprocessor or SpeechTextPreprocessor()
        self.last_spoken_text: Optional[str] = None
        self._client = None

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    @property
    def is_v4_model(self) -> bool:
        """Check whether configured model is an ElevenLabs v4 / dialogue model."""
        return "v4" in self.model_id.lower() or "dialogue" in self.model_id.lower()

    def _get_client(self):
        """Lazy instantiation of AsyncElevenLabs client."""
        if not self.api_key:
            raise TTSUnavailableError("ElevenLabs API key is not configured.")
        if self._client is None:
            from elevenlabs.client import AsyncElevenLabs
            self._client = AsyncElevenLabs(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _get_voice_settings(self):
        """
        Return voice settings supported by the configured model.
        v4 models only support stability and similarity_boost; style and speaker boost
        are omitted to adhere to the v4 model contract.
        """
        from elevenlabs import VoiceSettings
        if self.is_v4_model:
            return VoiceSettings(
                stability=self.stability,
                similarity_boost=self.similarity_boost,
            )
        return VoiceSettings(
            stability=self.stability,
            similarity_boost=self.similarity_boost,
            style=self.style,
            use_speaker_boost=self.use_speaker_boost,
        )

    def _get_dialogue_settings(self):
        """Settings model for Text-to-Dialogue endpoint."""
        from elevenlabs.types import ToDialogueSettingsResponseModel
        return ToDialogueSettingsResponseModel(
            stability=self.stability,
            similarity=self.similarity_boost,
        )

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """
        Resolve valid ElevenLabs voice ID.
        If voice is None or generic P-MAI default (e.g. 'natural-en-1', 'default'),
        fall back to configured self.voice_id.
        """
        if not voice or voice in ("default", "natural-en-1", "standard", "alloy", "echo", "fable", "onyx", "nova", "shimmer"):
            return self.voice_id
        return voice

    def _handle_exception(
        self,
        exc: Exception,
        voice_id: str = "",
        phase: str = "audio_generation",
    ) -> Exception:
        """Map provider exceptions to P-MAI TTS exceptions with structured, secret-free diagnostics."""
        from elevenlabs.core.api_error import ApiError

        status = getattr(exc, "status_code", None)
        body = getattr(exc, "body", {})
        provider_code = None
        msg = ""

        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, dict):
                msg = detail.get("message", "")
                provider_code = detail.get("code") or detail.get("status")
            elif isinstance(detail, str):
                msg = detail
        elif isinstance(body, str):
            msg = body

        used_voice = voice_id or self.voice_id
        operation = "text_to_dialogue" if self.is_v4_model else "text_to_speech"
        endpoint = "/v1/text-to-dialogue" if self.is_v4_model else f"/v1/text-to-speech/{used_voice}"

        if isinstance(exc, ApiError):
            if status in (401, 403):
                err = TTSUnavailableError(f"ElevenLabs authentication failed (invalid API key): {msg or 'Unauthorized'}")
            elif status == 402:
                err = TTSUnavailableError(f"ElevenLabs paid plan required: {msg or 'Payment required'}")
            elif status == 429:
                err = TTSUnavailableError(f"ElevenLabs rate limit exceeded (HTTP 429): {msg or 'Rate limited'}")
            elif status == 400:
                err = TTSUnavailableError(f"ElevenLabs bad request (HTTP 400): {msg or 'Invalid request parameters'}")
            elif status == 404:
                err = TTSUnavailableError(f"ElevenLabs resource not found (HTTP 404): {msg or 'Voice or model not found'}")
            elif status and status >= 500:
                err = TTSUnavailableError(f"ElevenLabs service unavailable (HTTP {status}): {msg or 'Internal server error'}")
            else:
                err = TTSUnavailableError(f"ElevenLabs API error (HTTP {status}): {msg or 'Unexpected API error'}")
        elif isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            err = TTSTimeoutError(f"ElevenLabs synthesis timed out after {self.timeout_seconds}s.")
        elif isinstance(exc, httpx.RequestError):
            err = TTSUnavailableError(f"ElevenLabs network error: {type(exc).__name__}")
        elif isinstance(exc, (TTSUnavailableError, TTSTimeoutError, TTSError)):
            err = exc
        else:
            err = TTSUnavailableError(f"ElevenLabs synthesis failed: {type(exc).__name__}: {exc}")

        # Attach actionable diagnostic metadata without secrets
        setattr(err, "original_exception_class", type(exc).__name__)
        setattr(err, "status_code", status)
        setattr(err, "provider_error_code", provider_code)
        setattr(err, "model_id", self.model_id)
        setattr(err, "voice_id", used_voice)
        setattr(err, "api_operation", operation)
        setattr(err, "endpoint_path", endpoint)
        setattr(err, "failure_phase", phase)
        setattr(err, "sanitized_message", msg or str(err))
        return err

    async def _fetch_audio_stream(self, client, text: str, voice_id: str, is_stream: bool):
        """
        Call appropriate endpoint (Text-to-Dialogue for v4 or Text-to-Speech)
        and return the audio chunk async generator.
        """
        if self.is_v4_model:
            from elevenlabs import DialogueInput
            inputs = [DialogueInput(text=text, voice_id=voice_id)]
            settings = self._get_dialogue_settings()
            if is_stream:
                res = client.text_to_dialogue.stream(
                    inputs=inputs,
                    model_id=self.model_id,
                    output_format=self.output_format,
                    settings=settings,
                )
            else:
                res = client.text_to_dialogue.convert(
                    inputs=inputs,
                    model_id=self.model_id,
                    output_format=self.output_format,
                    settings=settings,
                )
        else:
            vs = self._get_voice_settings()
            if is_stream:
                res = client.text_to_speech.stream(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model_id,
                    output_format=self.output_format,
                    voice_settings=vs,
                    optimize_streaming_latency=self.optimize_streaming_latency,
                )
            else:
                res = client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model_id,
                    output_format=self.output_format,
                    voice_settings=vs,
                    optimize_streaming_latency=self.optimize_streaming_latency,
                )

        if inspect.isawaitable(res):
            res = await res
        return res

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TTSResult:
        """
        Synthesize complete audio response.
        Supports segmented synthesis for long-form speech to adhere to provider bounds.
        Returns canonical 16kHz 16-bit Mono WAV.
        """
        if not text or not text.strip():
            return TTSResult(
                audio_bytes=b"",
                content_type="audio/wav",
                duration_seconds=0.0,
                provider=self.provider_name,
            )

        client = self._get_client()
        prep = self.preprocessor.preprocess(
            text,
            target_provider=self.provider_name,
            model_id=self.model_id,
            max_segment_chars=self.max_segment_chars,
        )
        segments = prep.segments if prep.segments else [prep.expressive_text]
        self.last_spoken_text = prep.expressive_text
        chosen_voice = self._resolve_voice(voice)

        t0 = time.perf_counter()
        pcm_chunks = []
        phase = "audio_generation"

        try:
            for s_idx, segment in enumerate(segments):
                seg_text = f"{prep.audio_tag} {segment}" if (s_idx == 0 and prep.audio_tag and not segment.startswith("[")) else segment
                try:
                    audio_stream = await self._fetch_audio_stream(client, seg_text, chosen_voice, is_stream=False)
                    async for chunk in audio_stream:
                        if chunk:
                            pcm_chunks.append(chunk)
                except Exception as first_exc:
                    # If library voice restricted on free tier (402), retry once with premade voice
                    err_msg = str(getattr(first_exc, "body", "")).lower()
                    status = getattr(first_exc, "status_code", None)
                    if status == 402 and ("library voice" in err_msg or "paid plan" in err_msg) and chosen_voice != PREMADE_FALLBACK_VOICE_ID:
                        logger.warning(
                            "Configured voice '%s' requires a paid plan on ElevenLabs API. Retrying with premade voice '%s'.",
                            chosen_voice,
                            PREMADE_FALLBACK_VOICE_ID,
                        )
                        chosen_voice = PREMADE_FALLBACK_VOICE_ID
                        audio_stream = await self._fetch_audio_stream(client, seg_text, chosen_voice, is_stream=False)
                        async for chunk in audio_stream:
                            if chunk:
                                pcm_chunks.append(chunk)
                    else:
                        raise first_exc

            phase = "audio_handling"
            raw_pcm = b"".join(pcm_chunks)
            if not raw_pcm:
                raise TTSError("ElevenLabs produced empty audio.")

            # Frame align raw PCM
            aligned_len = (len(raw_pcm) // 2) * 2
            raw_pcm = raw_pcm[:aligned_len]

            # Wrap canonical 16kHz mono PCM into valid WAV container
            wav_bytes = pcm_to_wav(
                raw_pcm,
                sample_rate=CanonicalAudioFormat.sample_rate,
                channels=CanonicalAudioFormat.channels,
                sample_width=CanonicalAudioFormat.sample_width,
            )
            duration_s = round(time.perf_counter() - t0, 3)

            return TTSResult(
                audio_bytes=wav_bytes,
                content_type="audio/wav",
                duration_seconds=duration_s,
                provider=self.provider_name,
            )
        except Exception as exc:
            raise self._handle_exception(exc, voice_id=chosen_voice, phase=phase) from exc

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Stream incremental synthesized audio chunks for low-latency playback.
        Synthesizes long texts sequentially across segmented paragraphs/sentences
        without dropping later paragraphs or exceeding API limits.
        Yields canonical 16kHz 16-bit Mono PCM chunks.
        """
        if not text or not text.strip():
            return

        client = self._get_client()
        prep = self.preprocessor.preprocess(
            text,
            target_provider=self.provider_name,
            model_id=self.model_id,
            max_segment_chars=self.max_segment_chars,
        )
        segments = prep.segments if prep.segments else [prep.expressive_text]
        self.last_spoken_text = prep.expressive_text
        chosen_voice = self._resolve_voice(voice)
        phase = "audio_generation"

        async def _iterate_stream(voice_to_use: str, text_payload: str):
            stream = await self._fetch_audio_stream(client, text_payload, voice_to_use, is_stream=True)
            async for chunk in stream:
                yield chunk

        try:
            active_voice = chosen_voice
            remainder = b""

            for s_idx, segment in enumerate(segments):
                seg_text = f"{prep.audio_tag} {segment}" if (s_idx == 0 and prep.audio_tag and not segment.startswith("[")) else segment

                try:
                    it = _iterate_stream(active_voice, seg_text)
                    first_chunk = await anext(it, None)
                except Exception as first_exc:
                    err_msg = str(getattr(first_exc, "body", "")).lower()
                    status = getattr(first_exc, "status_code", None)
                    if status == 402 and ("library voice" in err_msg or "paid plan" in err_msg) and active_voice != PREMADE_FALLBACK_VOICE_ID:
                        logger.warning(
                            "Configured voice '%s' requires a paid plan on ElevenLabs API. Retrying with premade voice '%s'.",
                            active_voice,
                            PREMADE_FALLBACK_VOICE_ID,
                        )
                        active_voice = PREMADE_FALLBACK_VOICE_ID
                        it = _iterate_stream(active_voice, seg_text)
                        first_chunk = await anext(it, None)
                    else:
                        raise first_exc

                phase = "audio_handling"
                if first_chunk:
                    data = remainder + first_chunk
                    if len(data) % 2 != 0:
                        remainder = data[-1:]
                        data = data[:-1]
                    else:
                        remainder = b""
                    if data:
                        yield TTSResult(
                            audio_bytes=data,
                            content_type="audio/wav",
                            duration_seconds=chunk_duration_seconds,
                            provider=self.provider_name,
                        )
                        await asyncio.sleep(0.001)

                async for chunk in it:
                    if not chunk:
                        continue
                    data = remainder + chunk
                    if len(data) % 2 != 0:
                        remainder = data[-1:]
                        data = data[:-1]
                    else:
                        remainder = b""
                    if data:
                        yield TTSResult(
                            audio_bytes=data,
                            content_type="audio/wav",
                            duration_seconds=chunk_duration_seconds,
                            provider=self.provider_name,
                        )
                        await asyncio.sleep(0.001)

            if remainder:
                # Pad odd byte with zero for frame alignment
                yield TTSResult(
                    audio_bytes=remainder + b"\x00",
                    content_type="audio/wav",
                    duration_seconds=chunk_duration_seconds,
                    provider=self.provider_name,
                )

        except Exception as exc:
            raise self._handle_exception(exc, voice_id=active_voice, phase=phase) from exc

