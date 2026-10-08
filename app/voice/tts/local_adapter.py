"""
P14 — Local-First Text-to-Speech Adapter using pyttsx3.
Offline TTS provider with lazy engine initialization, asyncio worker offloading,
canonical audio normalization, and streaming chunk generation.
"""
import asyncio
import io
import os
import tempfile
import time
from typing import AsyncGenerator

from app.voice.audio import AudioNormalizer, CanonicalAudioFormat, is_wav_container, pcm_to_wav, strip_wav_header
from app.voice.exceptions import TTSError, TTSTimeoutError, TTSUnavailableError
from app.voice.schemas import TTSResult
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.streaming import StreamingTextToSpeechProvider


class LocalTTSAdapter(StreamingTextToSpeechProvider):
    """
    Offline local Text-to-Speech provider wrapping pyttsx3.
    Never transmits speech text over the network.
    """

    def __init__(
        self,
        engine_name: str = "pyttsx3",
        default_voice: str | None = None,
        rate: int = 175,
        volume: float = 1.0,
        timeout_seconds: float = 30.0,
    ):
        self.engine_name = engine_name
        self.default_voice = default_voice
        self.rate = rate
        self.volume = volume
        self.timeout_seconds = timeout_seconds

        self._normalizer = AudioNormalizer(CanonicalAudioFormat.sample_rate)

    @property
    def provider_name(self) -> str:
        return f"local_{self.engine_name}"

    def __repr__(self) -> str:
        return f"LocalTTSAdapter(engine={self.engine_name}, voice={self.default_voice or 'default'}, rate={self.rate})"

    def _synthesize_file_sync(self, text: str, output_path: str, voice: str | None = None) -> None:
        """
        Synchronously synthesizes text to a WAV file using pyttsx3.
        Runs in a dedicated thread to ensure COM/event loops are isolated.
        """
        try:
            import pyttsx3  # type: ignore
        except ImportError as e:
            raise TTSUnavailableError(
                "Local TTS requires 'pyttsx3'. Install with 'pip install pyttsx3'."
            ) from e

        try:
            engine = pyttsx3.init()
            if self.rate:
                engine.setProperty("rate", self.rate)
            if self.volume:
                engine.setProperty("volume", self.volume)

            # Set voice if specified
            chosen_voice = voice or self.default_voice
            if chosen_voice:
                voices = engine.getProperty("voices")
                for v in voices:
                    if chosen_voice.lower() in getattr(v, "name", "").lower() or chosen_voice.lower() in getattr(v, "id", "").lower():
                        engine.setProperty("voice", v.id)
                        break

            engine.save_to_file(text, output_path)
            engine.runAndWait()
        except Exception as err:
            raise TTSError(f"Local pyttsx3 synthesis failed: {err}") from err

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str = "en",
    ) -> TTSResult:
        """Synthesize text to canonical 16kHz mono 16-bit WAV audio."""
        if not text or not text.strip():
            return TTSResult(audio_bytes=b"", content_type="audio/wav", duration_seconds=0.0, provider=self.provider_name)

        t0 = time.perf_counter()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        try:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._synthesize_file_sync, text, temp_file.name, voice),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise TTSTimeoutError(f"Local TTS synthesis timed out after {self.timeout_seconds}s.")

            if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
                raise TTSError("Local TTS produced empty audio file.")

            with open(temp_file.name, "rb") as f:
                raw_wav_bytes = f.read()

            # Normalize output to canonical 16kHz mono 16-bit WAV
            canonical_wav = self._normalizer.normalize_wav(raw_wav_bytes)
            duration_s = round(time.perf_counter() - t0, 3)

            return TTSResult(
                audio_bytes=canonical_wav,
                content_type="audio/wav",
                duration_seconds=duration_s,
                provider=self.provider_name,
            )
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        language: str = "en",
        chunk_duration_seconds: float = 0.5,
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Synthesizes text and yields continuous PCM chunks for streaming playback.
        Distinguishes network chunks from standalone container files.
        """
        if not text or not text.strip():
            return

        # Perform local synthesis
        result = await self.synthesize(text, voice=voice, language=language)
        if not result.audio_bytes:
            return

        # Extract continuous PCM data (strips standalone WAV container header)
        pcm_bytes = strip_wav_header(result.audio_bytes)
        bytes_per_second = CanonicalAudioFormat.bytes_per_second  # 32,000 bytes/sec
        chunk_size = int(bytes_per_second * chunk_duration_seconds)
        # Ensure frame boundary alignment (2 bytes per sample)
        chunk_size = (chunk_size // 2) * 2
        if chunk_size <= 0:
            chunk_size = 3200

        total_len = len(pcm_bytes)
        offset = 0

        while offset < total_len:
            end = min(offset + chunk_size, total_len)
            chunk = pcm_bytes[offset:end]
            offset = end

            yield TTSResult(
                audio_bytes=chunk,
                content_type="audio/wav",
                duration_seconds=chunk_duration_seconds,
                provider=self.provider_name,
            )
            # Brief cooperative yield to allow barge-in / cancellation
            await asyncio.sleep(0.001)
