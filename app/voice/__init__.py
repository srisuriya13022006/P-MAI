"""
P12 — Voice I/O Foundation for P-MAI.
Voice presentation and I/O abstractions.
"""
from app.voice.config import VoiceConfig
from app.voice.exceptions import (
    AudioTooLargeError,
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
    EmptyTranscriptionError,
    InvalidAudioError,
    STTError,
    STTTimeoutError,
    STTUnavailableError,
    TTSError,
    TTSTimeoutError,
    TTSUnavailableError,
    UnsupportedAudioFormatError,
    VoiceDisabledError,
    VoiceError,
)
from app.voice.pipeline import AudioValidator, VoicePipeline
from app.voice.schemas import (
    AudioValidationResult,
    STTResult,
    TTSResult,
    VoiceChatRequest,
    VoiceChatResponse,
    VoiceTimingMetadata,
)
from app.voice.events import RealtimeClientEvent, RealtimeEventType, RealtimeServerEvent
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.session import (
    RealtimeVoiceSession,
    SessionStatus,
    VALID_SESSION_TRANSITIONS,
    VoiceSessionManager,
)
from app.voice.stt import (
    MockSTTProvider,
    MockStreamingSTTProvider,
    SpeechToTextProvider,
    StreamingSpeechToTextProvider,
)
from app.voice.tts import (
    MockStreamingTTSProvider,
    MockTTSProvider,
    StreamingTextToSpeechProvider,
    TextToSpeechProvider,
    generate_mock_wav,
)
from app.voice.audio import (
    AudioNormalizer,
    BoundedAudioChunkBuffer,
    CanonicalAudioFormat,
    StreamingAudioAssembler,
    is_wav_container,
    pcm_to_wav,
    strip_wav_header,
)
from app.voice.formatter import VoicePresentationFormatter
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.stt.whisper_adapter import WhisperSTTAdapter
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.local_adapter import LocalTTSAdapter
from app.voice.tts.openai_adapter import OpenAITTSAdapter
from app.voice.vad import EnergyVADDetector, VADResult, VoiceActivityDetector

__all__ = [
    "VoiceConfig",
    "VoicePipeline",
    "AudioValidator",
    "AudioNormalizer",
    "CanonicalAudioFormat",
    "BoundedAudioChunkBuffer",
    "StreamingAudioAssembler",
    "is_wav_container",
    "strip_wav_header",
    "pcm_to_wav",
    "VoicePresentationFormatter",
    "RealtimeVoicePipeline",
    "VoiceSessionManager",
    "RealtimeVoiceSession",
    "SessionStatus",
    "VALID_SESSION_TRANSITIONS",
    "RealtimeEventType",
    "RealtimeClientEvent",
    "RealtimeServerEvent",
    "VoiceError",
    "VoiceDisabledError",
    "InvalidAudioError",
    "UnsupportedAudioFormatError",
    "AudioTooLargeError",
    "AudioTooLongError",
    "EmptyAudioError",
    "CorruptedAudioError",
    "STTError",
    "STTUnavailableError",
    "STTTimeoutError",
    "EmptyTranscriptionError",
    "TTSError",
    "TTSUnavailableError",
    "TTSTimeoutError",
    "AudioValidationResult",
    "STTResult",
    "TTSResult",
    "VoiceTimingMetadata",
    "VoiceChatRequest",
    "VoiceChatResponse",
    "SpeechToTextProvider",
    "MockSTTProvider",
    "StreamingSpeechToTextProvider",
    "MockStreamingSTTProvider",
    "WhisperSTTAdapter",
    "LocalWhisperSTTAdapter",
    "create_stt_provider",
    "TextToSpeechProvider",
    "MockTTSProvider",
    "StreamingTextToSpeechProvider",
    "MockStreamingTTSProvider",
    "OpenAITTSAdapter",
    "LocalTTSAdapter",
    "create_tts_provider",
    "generate_mock_wav",
    "VoiceActivityDetector",
    "EnergyVADDetector",
    "VADResult",
]
