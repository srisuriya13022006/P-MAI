"""
P12 — Voice I/O Foundation Exceptions.
Explicit error hierarchy for audio validation, STT, and TTS processing.
"""


class VoiceError(Exception):
    """Base exception for all voice I/O operations."""

    def __init__(self, message: str, error_code: str = "VOICE_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class VoiceDisabledError(VoiceError):
    """Raised when voice endpoints or features are invoked while voice is disabled."""

    def __init__(self, message: str = "Voice I/O is currently disabled."):
        super().__init__(message, error_code="VOICE_DISABLED")


class InvalidAudioError(VoiceError):
    """Base exception for audio format or content validation failures."""

    def __init__(self, message: str, error_code: str = "INVALID_AUDIO"):
        super().__init__(message, error_code=error_code)


class UnsupportedAudioFormatError(InvalidAudioError):
    """Raised when the audio format/mime-type is not supported."""

    def __init__(self, content_type: str, supported: list[str]):
        msg = f"Unsupported audio format '{content_type}'. Supported formats: {', '.join(supported)}."
        super().__init__(msg, error_code="UNSUPPORTED_AUDIO_FORMAT")
        self.content_type = content_type
        self.supported = supported


class AudioTooLargeError(InvalidAudioError):
    """Raised when the uploaded audio exceeds the configured maximum size in bytes."""

    def __init__(self, size_bytes: int, max_bytes: int):
        msg = f"Audio size {size_bytes} bytes exceeds maximum limit of {max_bytes} bytes."
        super().__init__(msg, error_code="AUDIO_TOO_LARGE")
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class AudioTooLongError(InvalidAudioError):
    """Raised when the audio duration exceeds the configured maximum duration."""

    def __init__(self, duration_seconds: float, max_seconds: float):
        msg = f"Audio duration {duration_seconds:.1f}s exceeds maximum limit of {max_seconds:.1f}s."
        super().__init__(msg, error_code="AUDIO_TOO_LONG")
        self.duration_seconds = duration_seconds
        self.max_seconds = max_seconds


class EmptyAudioError(InvalidAudioError):
    """Raised when the provided audio contains 0 bytes or only complete silence."""

    def __init__(self, message: str = "Audio input is empty or contains no detectable speech."):
        super().__init__(message, error_code="EMPTY_AUDIO")


class CorruptedAudioError(InvalidAudioError):
    """Raised when the audio header or content is malformed or corrupted."""

    def __init__(self, message: str = "Audio data is corrupted or cannot be parsed."):
        super().__init__(message, error_code="CORRUPTED_AUDIO")


class STTError(VoiceError):
    """Base exception for Speech-to-Text provider failures."""

    def __init__(self, message: str, error_code: str = "STT_ERROR"):
        super().__init__(message, error_code=error_code)


class STTUnavailableError(STTError):
    """Raised when the Speech-to-Text provider cannot be contacted or is unconfigured."""

    def __init__(self, message: str = "Speech-to-Text provider is unavailable."):
        super().__init__(message, error_code="STT_UNAVAILABLE")


class STTTimeoutError(STTError):
    """Raised when Speech-to-Text transcription times out."""

    def __init__(self, message: str = "Speech-to-Text transcription timed out."):
        super().__init__(message, error_code="STT_TIMEOUT")


class EmptyTranscriptionError(STTError):
    """Raised when transcription completes but returns empty or whitespace-only text."""

    def __init__(self, message: str = "Speech-to-Text produced an empty transcript."):
        super().__init__(message, error_code="EMPTY_TRANSCRIPTION")


class TTSError(VoiceError):
    """Base exception for Text-to-Speech provider failures."""

    def __init__(self, message: str, error_code: str = "TTS_ERROR"):
        super().__init__(message, error_code=error_code)


class TTSUnavailableError(TTSError):
    """Raised when the Text-to-Speech provider cannot be contacted or is unconfigured."""

    def __init__(self, message: str = "Text-to-Speech provider is unavailable."):
        super().__init__(message, error_code="TTS_UNAVAILABLE")


class TTSTimeoutError(TTSError):
    """Raised when Text-to-Speech synthesis times out."""

    def __init__(self, message: str = "Text-to-Speech synthesis timed out."):
        super().__init__(message, error_code="TTS_TIMEOUT")


class RealtimeVoiceError(VoiceError):
    """Base exception for realtime voice session and streaming operations."""

    def __init__(self, message: str, error_code: str = "REALTIME_VOICE_ERROR"):
        super().__init__(message, error_code=error_code)


class InvalidSessionStateTransitionError(RealtimeVoiceError):
    """Raised when an illegal voice session status transition is attempted."""

    def __init__(self, current_status: str, target_status: str):
        msg = f"Invalid voice session transition from '{current_status}' to '{target_status}'."
        super().__init__(msg, error_code="INVALID_SESSION_TRANSITION")
        self.current_status = current_status
        self.target_status = target_status


class SessionLimitExceededError(RealtimeVoiceError):
    """Raised when concurrent session limit is exceeded."""

    def __init__(self, current: int, limit: int):
        msg = f"Maximum concurrent voice sessions reached ({current}/{limit})."
        super().__init__(msg, error_code="SESSION_LIMIT_EXCEEDED")
        self.current = current
        self.limit = limit


class SessionExpiredError(RealtimeVoiceError):
    """Raised when a voice session exceeds maximum allowed lifetime."""

    def __init__(self, duration_s: float, max_s: float):
        msg = f"Voice session expired after {duration_s:.1f}s (max {max_s:.1f}s)."
        super().__init__(msg, error_code="SESSION_EXPIRED")
        self.duration_s = duration_s
        self.max_s = max_s


class SessionInactivityTimeoutError(RealtimeVoiceError):
    """Raised when a voice session is inactive beyond the configured timeout."""

    def __init__(self, idle_s: float, max_s: float):
        msg = f"Voice session timed out due to inactivity for {idle_s:.1f}s (max {max_s:.1f}s)."
        super().__init__(msg, error_code="SESSION_INACTIVITY_TIMEOUT")
        self.idle_s = idle_s
        self.max_s = max_s


class UnauthorizedSessionError(RealtimeVoiceError):
    """Raised when accessing an unknown, unauthorized, or mismatched session."""

    def __init__(self, message: str = "Unauthorized or non-existent voice session."):
        super().__init__(message, error_code="UNAUTHORIZED_SESSION")


class BackpressureError(RealtimeVoiceError):
    """Raised when audio buffer or queue exceeds backpressure limits."""

    def __init__(self, buffered_bytes: int, max_bytes: int):
        msg = f"Audio buffer backpressure limit exceeded ({buffered_bytes} > {max_bytes} bytes)."
        super().__init__(msg, error_code="BACKPRESSURE_LIMIT_EXCEEDED")
        self.buffered_bytes = buffered_bytes
        self.max_bytes = max_bytes


class MalformedRealtimeEventError(RealtimeVoiceError):
    """Raised when an incoming realtime event is malformed or invalid."""

    def __init__(self, message: str = "Malformed or unsupported realtime voice event."):
        super().__init__(message, error_code="MALFORMED_REALTIME_EVENT")


class SessionClosedError(RealtimeVoiceError):
    """Raised when attempting actions on an already closed voice session."""

    def __init__(self, session_id: str):
        super().__init__(f"Voice session '{session_id}' is closed.", error_code="SESSION_CLOSED")
        self.session_id = session_id
