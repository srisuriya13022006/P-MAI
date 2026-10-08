"""
P13 — Audio Normalization, Canonical Format & Realtime Chunk Handling.
Enforces canonical 16kHz 16-bit Mono PCM audio, channel downmixing, resampling,
frame alignment, duplicate chunk suppression, and bounded buffering.
"""
from dataclasses import dataclass
import hashlib
import io
import math
import struct
import wave
from app.voice.exceptions import (
    AudioTooLargeError,
    BackpressureError,
    CorruptedAudioError,
    EmptyAudioError,
    InvalidAudioError,
)


@dataclass(frozen=True)
class CanonicalAudioFormat:
    """Canonical realtime audio specification for P-MAI voice pipeline."""

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM (2 bytes per sample)
    bytes_per_second: int = 32000  # 16000 * 1 * 2


class AudioNormalizer:
    """
    Normalizes arbitrary audio streams into the canonical format:
    16-bit Little-Endian Signed PCM, Mono (1 channel), 16000 Hz.
    """

    def __init__(self, target_rate: int = 16000):
        self.target_rate = target_rate

    def normalize_pcm(
        self,
        pcm_bytes: bytes,
        in_sample_rate: int = 16000,
        in_channels: int = 1,
        in_sample_width: int = 2,
    ) -> bytes:
        """
        Normalize raw PCM bytes to canonical (16kHz, mono, 16-bit signed).
        Handles alignment, stereo-to-mono downmixing, and sample rate conversion.
        """
        if not pcm_bytes:
            return b""

        # 1. Align frame boundaries
        bytes_per_frame = in_channels * in_sample_width
        aligned_len = (len(pcm_bytes) // bytes_per_frame) * bytes_per_frame
        if aligned_len == 0:
            return b""
        data = pcm_bytes[:aligned_len]

        num_frames = aligned_len // bytes_per_frame

        # 2. Extract samples as integers
        if in_sample_width == 2:
            fmt = f"<{num_frames * in_channels}h"
            raw_samples = struct.unpack(fmt, data)
        elif in_sample_width == 1:
            raw_samples = [((b - 128) << 8) for b in data]
        else:
            # Fallback 16-bit assumption
            raw_samples = struct.unpack(f"<{len(data) // 2}h", data[: (len(data) // 2) * 2])

        # 3. Downmix multi-channel to mono
        if in_channels == 2:
            mono_samples = []
            for i in range(0, len(raw_samples), 2):
                left = raw_samples[i]
                right = raw_samples[i + 1]
                avg = (left + right) // 2
                mono_samples.append(avg)
        elif in_channels > 2:
            mono_samples = []
            for i in range(0, len(raw_samples), in_channels):
                avg = sum(raw_samples[i : i + in_channels]) // in_channels
                mono_samples.append(avg)
        else:
            mono_samples = list(raw_samples)

        # 4. Resample if necessary (Linear Interpolation)
        if in_sample_rate != self.target_rate and len(mono_samples) > 1:
            resampled = self._resample_linear(mono_samples, in_sample_rate, self.target_rate)
        else:
            resampled = mono_samples

        # 5. Pack back to 16-bit signed Little-Endian PCM
        out_buf = bytearray()
        for sample in resampled:
            clamped = max(-32768, min(32767, int(sample)))
            out_buf.extend(struct.pack("<h", clamped))

        return bytes(out_buf)

    def normalize_wav(self, wav_bytes: bytes) -> bytes:
        """
        Parse WAV container and normalize audio frames into canonical WAV.
        """
        if not wav_bytes.startswith(b"RIFF"):
            raise CorruptedAudioError("Invalid WAV header.")

        try:
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    n_channels = wf.getnchannels()
                    samp_width = wf.getsampwidth()
                    rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)

            canonical_pcm = self.normalize_pcm(
                frames,
                in_sample_rate=rate,
                in_channels=n_channels,
                in_sample_width=samp_width,
            )

            # Reconstruct canonical WAV header
            out_io = io.BytesIO()
            with wave.open(out_io, "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(2)
                out_wf.setframerate(self.target_rate)
                out_wf.writeframes(canonical_pcm)
            return out_io.getvalue()
        except wave.Error as err:
            raise CorruptedAudioError(f"Malformed WAV structure: {err}")

    def _resample_linear(self, samples: list[int], in_rate: int, out_rate: int) -> list[int]:
        """High-speed linear interpolation for sample rate matching."""
        if not samples:
            return []
        ratio = float(in_rate) / float(out_rate)
        target_len = int(len(samples) / ratio)
        resampled = []

        for i in range(target_len):
            src_idx = i * ratio
            idx0 = int(src_idx)
            idx1 = min(idx0 + 1, len(samples) - 1)
            frac = src_idx - idx0
            val = (1.0 - frac) * samples[idx0] + frac * samples[idx1]
            resampled.append(int(val))

        return resampled


class BoundedAudioChunkBuffer:
    """
    Bounded buffer for streaming audio chunks with duplicate suppression,
    max capacity limits, and packet boundary tracking.
    """

    def __init__(self, max_bytes: int = 2 * 1024 * 1024, max_history: int = 100):
        self.max_bytes = max_bytes
        self.max_history = max_history
        self._buffer = bytearray()
        self._seen_hashes: set[str] = set()
        self._hash_queue: list[str] = []
        self._dropped_duplicates = 0

    @property
    def current_size(self) -> int:
        return len(self._buffer)

    def append_chunk(self, chunk: bytes, allow_duplicate: bool = False) -> bool:
        """
        Append an audio chunk. Returns False if dropped as duplicate.
        Raises BackpressureError if capacity would be exceeded.
        """
        if not chunk:
            return True

        if len(self._buffer) + len(chunk) > self.max_bytes:
            raise BackpressureError(len(self._buffer) + len(chunk), self.max_bytes)

        if not allow_duplicate:
            chunk_hash = hashlib.md5(chunk).hexdigest()
            if chunk_hash in self._seen_hashes:
                self._dropped_duplicates += 1
                return False  # Silently suppress duplicate frame

            self._seen_hashes.add(chunk_hash)
            self._hash_queue.append(chunk_hash)
            if len(self._hash_queue) > self.max_history:
                oldest = self._hash_queue.pop(0)
                self._seen_hashes.discard(oldest)

        self._buffer.extend(chunk)
        return True

    def get_and_clear(self) -> bytes:
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def clear(self) -> None:
        self._buffer.clear()
        self._seen_hashes.clear()
        self._hash_queue.clear()


def is_wav_container(data: bytes) -> bool:
    """Check if data starts with RIFF WAV container header."""
    return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"


def strip_wav_header(data: bytes) -> bytes:
    """
    Extract raw PCM frames from a WAV container if present.
    If already raw PCM, returns data unchanged.
    """
    if not is_wav_container(data):
        return data
    try:
        with io.BytesIO(data) as buf:
            with wave.open(buf, "rb") as wf:
                return wf.readframes(wf.getnframes())
    except wave.Error:
        # Fallback to standard 44-byte standard header offset
        return data[44:] if len(data) > 44 else data


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """
    Package raw PCM frames into a standalone, independently playable WAV container.
    """
    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return out_io.getvalue()


class StreamingAudioAssembler:
    """
    Assembles streaming audio chunks into a continuous PCM audio stream.
    Correctly distinguishes between network chunks (parts of continuous audio)
    and standalone containerized audio files (WAV).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._pcm_buffer = bytearray()
        self._chunk_count = 0

    @property
    def total_bytes(self) -> int:
        return len(self._pcm_buffer)

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    def append_chunk(self, chunk: bytes) -> None:
        """
        Append network chunk. If the chunk carries a WAV container header,
        strips the header so the stream remains unbroken continuous PCM.
        """
        if not chunk:
            return
        pcm_data = strip_wav_header(chunk) if is_wav_container(chunk) else chunk
        self._pcm_buffer.extend(pcm_data)
        self._chunk_count += 1

    def get_continuous_pcm(self) -> bytes:
        """Return the complete accumulated continuous PCM audio bytes."""
        return bytes(self._pcm_buffer)

    def get_standalone_wav(self) -> bytes:
        """Return the accumulated audio wrapped in a valid standalone WAV container."""
        return pcm_to_wav(
            bytes(self._pcm_buffer),
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )

    def clear(self) -> None:
        """Reset the stream assembler buffer."""
        self._pcm_buffer.clear()
        self._chunk_count = 0
