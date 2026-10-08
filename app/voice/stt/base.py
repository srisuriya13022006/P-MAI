"""
P12 — Speech-to-Text Provider Base Interface.
Defines replaceable provider abstraction for audio transcription.
"""
from abc import ABC, abstractmethod
from app.voice.schemas import STTResult


class SpeechToTextProvider(ABC):
    """
    Abstract interface for Speech-to-Text (STT) providers.
    All concrete STT integrations (e.g. Whisper, Google Cloud Speech, Mock)
    must implement this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name/identifier of the STT provider."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """
        Transcribe audio byte stream to plain text.

        Args:
            audio_bytes: Raw audio binary data.
            content_type: MIME content type of the audio.
            language: Optional BCP-47 / ISO language code.

        Returns:
            STTResult containing transcript, detected/used language, confidence, and provider.

        Raises:
            STTUnavailableError: If the provider is unreachable or unconfigured.
            STTTimeoutError: If transcription times out.
            EmptyTranscriptionError: If no speech was transcribed.
        """
        pass
