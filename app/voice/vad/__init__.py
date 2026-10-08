"""
P12 — Voice Activity Detection (VAD) module.
"""
from app.voice.vad.base import VADResult, VoiceActivityDetector
from app.voice.vad.detector import EnergyVADDetector

__all__ = ["VADResult", "VoiceActivityDetector", "EnergyVADDetector"]
