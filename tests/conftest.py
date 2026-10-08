"""
Pytest configuration for P-MAI test suite.
Ensures deterministic mock voice providers are used for automated tests
unless individual tests explicitly test local or cloud providers.
"""
import os

# Set mock provider defaults for automated test suite runs
os.environ.setdefault("VOICE_STT_PROVIDER", "mock")
os.environ.setdefault("VOICE_TTS_PROVIDER", "mock")
