"""
P14 — Local-First Speech-to-Text Adapter using Local Whisper.
Supports offline transcription via faster-whisper (or openai-whisper) with lazy model initialization,
asyncio worker offloading, canonical audio normalization, and streaming frame ingestion.
"""
import asyncio
import io
import os
import tempfile
import time
from typing import Any

from app.voice.audio import AudioNormalizer, CanonicalAudioFormat, StreamingAudioAssembler, is_wav_container, pcm_to_wav
from app.voice.exceptions import (
    EmptyAudioError,
    STTError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.streaming import StreamingSpeechToTextProvider


class LocalWhisperSTTAdapter(StreamingSpeechToTextProvider):
    """
    Offline local Speech-to-Text provider wrapping faster-whisper (or openai-whisper).
    Never transmits audio over the network.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        timeout_seconds: float = 30.0,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.timeout_seconds = timeout_seconds

        self._model: Any = None
        self._backend: str | None = None
        self._init_lock = asyncio.Lock()
        self._normalizer = AudioNormalizer(CanonicalAudioFormat.sample_rate)
        self._stream_assembler = StreamingAudioAssembler()
        self._last_partial_transcript = ""

    @property
    def provider_name(self) -> str:
        return f"local_whisper_{self.model_name}"

    def __repr__(self) -> str:
        return f"LocalWhisperSTTAdapter(model={self.model_name}, device={self.device}, compute_type={self.compute_type})"

    def _ensure_model_loaded(self) -> Any:
        """
        Synchronously loads the local Whisper model in a worker thread.
        Prefers faster-whisper; falls back to openai-whisper.
        """
        if self._model is not None:
            return self._model

        # 1. Try faster-whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._backend = "faster-whisper"
            return self._model
        except ImportError:
            pass
        except Exception as e:
            raise STTUnavailableError(
                f"Failed to initialize local faster-whisper model '{self.model_name}': {e}"
            ) from e

        # 2. Try openai-whisper
        try:
            import whisper  # type: ignore

            self._model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            return self._model
        except ImportError:
            pass
        except Exception as e:
            raise STTUnavailableError(
                f"Failed to initialize local openai-whisper model '{self.model_name}': {e}"
            ) from e

        # Neither library is installed
        raise STTUnavailableError(
            "Local Whisper STT requires 'faster-whisper' or 'openai-whisper'. "
            "Install with 'pip install faster-whisper'."
        )

    def _transcribe_file_sync(self, wav_path: str) -> STTResult:
        """Synchronous execution of model transcription on disk file."""
        model = self._ensure_model_loaded()
        t0 = time.perf_counter()

        if self._backend == "faster-whisper":
            segments, info = model.transcribe(
                wav_path,
                language=self.language,
                beam_size=1,
                vad_filter=True,
            )
            transcript_parts = [s.text.strip() for s in segments]
            full_text = " ".join(filter(None, transcript_parts)).strip()
            confidence = float(getattr(info, "avg_logprob", 0.0))
            lang = getattr(info, "language", self.language)
        elif self._backend == "openai-whisper":
            result = model.transcribe(
                wav_path,
                language=self.language,
                fp16=False,
            )
            full_text = result.get("text", "").strip()
            confidence = 0.9 if full_text else 0.0
            lang = result.get("language", self.language)
        else:
            raise STTError("No local Whisper backend available.")

        duration_seconds = round(time.perf_counter() - t0, 3)
        return STTResult(
            transcript=full_text,
            confidence=max(0.0, min(1.0, 1.0 + confidence if confidence < 0 else confidence)),
            duration_seconds=duration_seconds,
            language=lang,
            provider=self.provider_name,
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
    ) -> STTResult:
        """Transcribe raw audio bytes using the local Whisper model."""
        if not audio_bytes or len(audio_bytes) < 44:
            return STTResult(transcript="", confidence=0.0, duration_seconds=0.0, language=self.language, provider=self.provider_name)

        # Normalize to canonical WAV
        if is_wav_container(audio_bytes):
            try:
                canonical_wav = self._normalizer.normalize_wav(audio_bytes)
            except Exception:
                canonical_wav = audio_bytes
        else:
            canonical_pcm = self._normalizer.normalize_pcm(audio_bytes)
            canonical_wav = pcm_to_wav(canonical_pcm)

        # Write to temporary file for Whisper ingestion
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_file.write(canonical_wav)
            temp_file.flush()
            temp_file.close()

            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._transcribe_file_sync, temp_file.name),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise STTTimeoutError(f"Local Whisper transcription timed out after {self.timeout_seconds}s.")
            except Exception as e:
                if isinstance(e, (STTUnavailableError, STTTimeoutError, STTError)):
                    raise
                raise STTError(f"Local Whisper transcription failed: {e}") from e
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # StreamingSpeechToTextProvider Contract
    # -------------------------------------------------------------------------

    async def feed_audio_chunk(
        self,
        chunk_bytes: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        """
        Accumulates incoming realtime audio chunks.
        Emits periodic partial transcripts when enough audio is accumulated.
        """
        if not chunk_bytes:
            return []

        self._stream_assembler.append_chunk(chunk_bytes)
        current_len = self._stream_assembler.total_bytes

        # Generate partial updates every ~0.5s of audio (16,000 bytes)
        if current_len >= 16000 and (current_len % 16000 < len(chunk_bytes)):
            try:
                # Transcribe current slice for partial transcript
                pcm_data = self._stream_assembler.get_continuous_pcm()
                wav_data = pcm_to_wav(pcm_data)
                res = await self.transcribe(wav_data)
                if res.transcript and res.transcript != self._last_partial_transcript:
                    self._last_partial_transcript = res.transcript
                    return [res]
            except Exception:
                pass

        return []

    async def finalize_stream(self) -> STTResult:
        """Finalize accumulated stream and return authoritative final transcript."""
        pcm_data = self._stream_assembler.get_continuous_pcm()
        if not pcm_data:
            self.reset()
            return STTResult(transcript="", confidence=0.0, duration_seconds=0.0, language=self.language, provider=self.provider_name)

        wav_data = pcm_to_wav(pcm_data)
        try:
            result = await self.transcribe(wav_data)
            return result
        finally:
            self.reset()

    def reset(self) -> None:
        """Reset internal streaming buffer."""
        self._stream_assembler.clear()
        self._last_partial_transcript = ""
