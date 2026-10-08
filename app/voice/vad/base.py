"""
P12 — Voice Activity Detection Base Interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VADResult:
    """Outcome of Voice Activity Detection analysis."""

    speech_detected: bool
    duration_seconds: float
    speech_energy_db: float | None = None
    silence_ratio: float = 0.0


class VoiceActivityDetector(ABC):
    """Abstract interface for Voice Activity Detection."""

    @abstractmethod
    def detect(self, audio_bytes: bytes, content_type: str = "audio/wav") -> VADResult:
        """
        Analyze audio input for speech vs silence within bounded duration limits.
        """
        pass
