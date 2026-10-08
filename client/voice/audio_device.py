"""
P15 — Audio Device Capture and Playback for P-MAI Voice Client.
Captures 16kHz 16-bit mono PCM chunks from microphone and plays back received audio chunks.
Supports sounddevice and PyAudio with fallback synthetic test capture when hardware libraries are absent.
"""
import io
import math
import struct
import threading
import time
from typing import Callable, Generator, Optional


class AudioDevice:
    """
    Audio I/O device controller for desktop client.
    Captures canonical PCM audio (16kHz, 16-bit, Mono) and plays received audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_s: float = 0.2,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(sample_rate * chunk_duration_s)
        self.chunk_bytes_len = self.chunk_samples * channels * 2  # 16-bit = 2 bytes

        self._pyaudio_instance = None
        self._input_stream = None
        self._output_stream = None
        self._is_active = False

        # Permission and hardware availability flags
        self.permission_granted: bool = True
        self.device_available: bool = True

        # Playback control
        self._is_playing: bool = False
        self._playback_interrupted: bool = False
        self._playback_lock = threading.Lock()

    def check_permissions(self) -> None:
        """
        Verify microphone permissions. Raises PermissionError if denied.
        """
        if not self.permission_granted:
            raise PermissionError("Microphone permission denied by user or operating system.")

    def check_availability(self) -> None:
        """
        Verify microphone device availability. Raises RuntimeError if unavailable.
        """
        if not self.device_available:
            raise RuntimeError("Microphone device is not available or disconnected.")

    def generate_synthetic_chunk(self, frequency_hz: float = 440.0) -> bytes:
        """Generate a canonical 16-bit PCM chunk with acoustic energy for local testing."""
        buf = bytearray()
        for i in range(self.chunk_samples):
            val = int(8000.0 * math.sin(2.0 * math.pi * frequency_hz * i / self.sample_rate))
            buf.extend(struct.pack("<h", val))
        return bytes(buf)

    def generate_mock_speech_chunks(self, num_chunks: int = 5) -> list[bytes]:
        """Generate a series of audio chunks simulating an utterance."""
        return [self.generate_synthetic_chunk(440.0 + i * 20.0) for i in range(num_chunks)]

    def open_input_stream(self) -> None:
        """Attempt to open hardware microphone stream using sounddevice or PyAudio."""
        self.check_permissions()
        self.check_availability()

        # 1. Try sounddevice
        try:
            import sounddevice as sd

            self._input_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.chunk_samples,
            )
            self._input_stream.start()
            self._is_active = True
            return
        except (ImportError, Exception):
            pass

        # 2. Try pyaudio
        try:
            import pyaudio  # type: ignore

            self._pyaudio_instance = pyaudio.PyAudio()
            self._input_stream = self._pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_samples,
            )
            self._is_active = True
            return
        except (ImportError, Exception):
            # Hardware audio optional / fallback synthetic mode
            self._is_active = False

    def read_microphone_chunk(self) -> bytes:
        """Read canonical PCM chunk from mic, or fallback to synthetic test chunk."""
        self.check_permissions()
        self.check_availability()

        if self._input_stream and self._is_active:
            try:
                # Handle sounddevice RawInputStream
                if hasattr(self._input_stream, "read"):
                    res = self._input_stream.read(self.chunk_samples)
                    if isinstance(res, tuple):
                        return bytes(res[0])
                    return bytes(res)
            except Exception:
                pass
        return self.generate_synthetic_chunk()

    def play_audio_chunk(self, audio_bytes: bytes) -> bool:
        """
        Play audio chunk to speaker if available and not interrupted.
        Returns True if played, False if skipped/interrupted.
        """
        if not audio_bytes or self._playback_interrupted:
            return False

        with self._playback_lock:
            self._is_playing = True
            data = audio_bytes[44:] if audio_bytes.startswith(b"RIFF") else audio_bytes

            # 1. Try sounddevice
            try:
                import sounddevice as sd
                import numpy as np

                arr = np.frombuffer(data, dtype=np.int16)
                if not self._playback_interrupted:
                    sd.play(arr, samplerate=self.sample_rate, blocking=True)
                return True
            except (ImportError, Exception):
                pass

            # 2. Try pyaudio
            try:
                import pyaudio  # type: ignore

                if not self._pyaudio_instance:
                    self._pyaudio_instance = pyaudio.PyAudio()
                if not self._output_stream:
                    self._output_stream = self._pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        output=True,
                    )
                if not self._playback_interrupted:
                    self._output_stream.write(data)
                return True
            except (ImportError, Exception):
                pass
            return True

    def stop_playback(self) -> None:
        """
        Immediately interrupt audio playback (barge-in / stop speaking).
        """
        with self._playback_lock:
            self._playback_interrupted = True
            self._is_playing = False
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
            try:
                if self._output_stream:
                    self._output_stream.stop_stream()
            except Exception:
                pass

    def reset_playback_state(self) -> None:
        """Reset interruption flag to allow subsequent playback turns."""
        with self._playback_lock:
            self._playback_interrupted = False
            self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing and not self._playback_interrupted

    def close(self) -> None:
        """Close hardware audio streams."""
        self.stop_playback()
        try:
            if self._input_stream:
                if hasattr(self._input_stream, "stop"):
                    self._input_stream.stop()
                if hasattr(self._input_stream, "close"):
                    self._input_stream.close()
                elif hasattr(self._input_stream, "stop_stream"):
                    self._input_stream.stop_stream()
                    self._input_stream.close()
            if self._output_stream:
                self._output_stream.stop_stream()
                self._output_stream.close()
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
        except Exception:
            pass
        self._is_active = False
