"""
P12 — Text-to-Speech Provider Base Interface.
Defines replaceable provider abstraction for speech synthesis.
"""
from abc import ABC, abstractmethod
from app.voice.schemas import TTSResult


class TextToSpeechProvider(ABC):
    """
    Abstract interface for Text-to-Speech (TTS) providers.
    All concrete TTS integrations (e.g. ElevenLabs, Google TTS, Mock)
    must implement this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name/identifier of the TTS provider."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> TTSResult:
        """
        Synthesize plain text into audio bytes.

        Args:
            text: Grounded text response to speak.
            voice: Optional voice character / model identifier.
            language: Optional language code.

        Returns:
            TTSResult containing audio_bytes, content_type, duration, and provider.

        Raises:
            TTSUnavailableError: If the provider is unreachable or unconfigured.
            TTSTimeoutError: If synthesis times out.
        """
        pass
