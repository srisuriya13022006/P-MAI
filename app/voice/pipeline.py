"""
P12 — Voice I/O Pipeline & Audio Validator.
Coordinates Audio Validation -> VAD -> STT -> MAI Brain -> TTS -> Voice Response.
Enforces that voice is an I/O layer and not a second brain.
"""
import base64
import binascii
import io
import time
import wave
from app.agent.orchestrator import MAIOrchestrator
from app.services.chat_service import ChatService
from app.voice.config import VoiceConfig
from app.voice.exceptions import (
    AudioTooLargeError,
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
    EmptyTranscriptionError,
    STTError,
    STTUnavailableError,
    TTSError,
    TTSUnavailableError,
    UnsupportedAudioFormatError,
    VoiceDisabledError,
)
from app.voice.schemas import (
    AudioValidationResult,
    STTResult,
    TTSResult,
    VoiceChatRequest,
    VoiceChatResponse,
    VoiceTimingMetadata,
)
from app.voice.formatter import VoicePresentationFormatter
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.mock import MockSTTProvider
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.mock import MockTTSProvider
from app.voice.vad.base import VoiceActivityDetector
from app.voice.vad.detector import EnergyVADDetector


class AudioValidator:
    """
    Validates audio payload constraints, content types, and format headers.
    """

    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()

    def validate(self, audio_bytes: bytes, content_type: str = "audio/wav") -> AudioValidationResult:
        """
        Validate audio bytes and content type against limits and headers.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise EmptyAudioError("Audio byte stream is empty (0 bytes).")

        size_bytes = len(audio_bytes)
        if size_bytes > self.config.max_audio_size_bytes:
            raise AudioTooLargeError(size_bytes, self.config.max_audio_size_bytes)

        normalized_ct = content_type.lower().strip()
        supported = [fmt.lower().strip() for fmt in self.config.supported_formats]

        if normalized_ct not in supported:
            raise UnsupportedAudioFormatError(normalized_ct, self.config.supported_formats)

        estimated_duration = None

        if normalized_ct == "audio/wav":
            if not audio_bytes.startswith(b"RIFF") or len(audio_bytes) < 44:
                raise CorruptedAudioError("Audio header does not contain a valid RIFF/WAV signature.")
            if audio_bytes[8:12] != b"WAVE":
                raise CorruptedAudioError("Audio header does not contain a valid WAVE format marker.")

            try:
                with io.BytesIO(audio_bytes) as buf:
                    with wave.open(buf, "rb") as wf:
                        framerate = wf.getframerate()
                        nframes = wf.getnframes()
                        if framerate <= 0:
                            framerate = 16000
                        duration = nframes / float(framerate)
                        estimated_duration = duration

                        if duration > self.config.max_duration_seconds:
                            raise AudioTooLongError(duration, self.config.max_duration_seconds)
            except wave.Error as we:
                raise CorruptedAudioError(f"Corrupted or malformed WAV structure: {we}")

        elif normalized_ct in ("audio/mpeg", "audio/mp3"):
            if not (
                audio_bytes.startswith(b"ID3")
                or audio_bytes.startswith(b"\xff\xfb")
                or audio_bytes.startswith(b"\xff\xf3")
                or audio_bytes.startswith(b"\xff\xf2")
            ):
                raise CorruptedAudioError("Audio header does not contain a valid MP3/ID3 signature.")
            # 128 kbps heuristic: 16000 bytes/sec
            estimated_duration = size_bytes / 16000.0
            if estimated_duration > self.config.max_duration_seconds:
                raise AudioTooLongError(estimated_duration, self.config.max_duration_seconds)

        elif normalized_ct == "audio/ogg":
            if not audio_bytes.startswith(b"OggS"):
                raise CorruptedAudioError("Audio header does not contain a valid OggS signature.")
            estimated_duration = size_bytes / 16000.0
            if estimated_duration > self.config.max_duration_seconds:
                raise AudioTooLongError(estimated_duration, self.config.max_duration_seconds)

        elif normalized_ct == "audio/flac":
            if not audio_bytes.startswith(b"fLaC"):
                raise CorruptedAudioError("Audio header does not contain a valid FLAC signature.")
            estimated_duration = size_bytes / 32000.0
            if estimated_duration > self.config.max_duration_seconds:
                raise AudioTooLongError(estimated_duration, self.config.max_duration_seconds)

        return AudioValidationResult(
            is_valid=True,
            content_type=normalized_ct,
            size_bytes=size_bytes,
            estimated_duration_seconds=estimated_duration,
        )


class VoicePipeline:
    """
    Authoritative Voice I/O Pipeline.
    Strictly acts as a presentation layer:
    Voice Input -> STT -> Existing MAI Chat / Orchestrator -> TTS -> Voice Output.
    """

    def __init__(
        self,
        chat_service: ChatService | None = None,
        orchestrator: MAIOrchestrator | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        vad_detector: VoiceActivityDetector | None = None,
        config: VoiceConfig | None = None,
    ):
        self.config = config or VoiceConfig.from_settings()
        self.chat_service = chat_service
        self.orchestrator = orchestrator
        self.validator = AudioValidator(self.config)
        self.formatter = VoicePresentationFormatter()

        # Initialize providers
        self.stt_provider = stt_provider or self._resolve_stt_provider()
        self.tts_provider = tts_provider or self._resolve_tts_provider()
        self.vad_detector = vad_detector or EnergyVADDetector(
            silence_threshold_db=self.config.vad_silence_threshold_db,
            max_duration_seconds=self.config.max_duration_seconds,
        )

    def _resolve_stt_provider(self) -> SpeechToTextProvider:
        return create_stt_provider(self.config)

    def _resolve_tts_provider(self) -> TextToSpeechProvider:
        return create_tts_provider(self.config)

    async def process_voice(self, request: VoiceChatRequest) -> VoiceChatResponse:
        """
        Execute end-to-end voice turnaround.

        Sequence:
        1. Check voice enabled
        2. Decode & validate audio
        3. Run VAD (Voice Activity Detection)
        4. Transcribe via STT
        5. Invoke authoritative MAI brain (reuses existing P0-P11 pipeline & P8 task state)
        6. Synthesize grounded answer via TTS (TTS failure is isolated and non-fatal)
        7. Return structured VoiceChatResponse (internal IDs & traces strictly hidden)
        """
        total_start = time.perf_counter()

        if not self.config.enabled:
            raise VoiceDisabledError("Voice I/O is disabled in configuration.")

        # 1. Base64 decode
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except (binascii.Error, ValueError) as err:
            raise CorruptedAudioError(f"Invalid base64 audio payload: {err}")

        # 2. Audio validation
        self.validator.validate(audio_bytes, request.content_type)

        # 3. VAD
        if self.vad_detector:
            vad_res = self.vad_detector.detect(audio_bytes, request.content_type)
            if not vad_res.speech_detected:
                raise EmptyAudioError("Audio contains no detectable speech.")

        # 4. STT transcription
        stt_start = time.perf_counter()
        stt_result = await self.stt_provider.transcribe(
            audio_bytes=audio_bytes,
            content_type=request.content_type,
            language=request.language or self.config.default_language,
        )
        stt_latency_ms = round((time.perf_counter() - stt_start) * 1000.0, 2)

        if not stt_result.transcript or not stt_result.transcript.strip():
            raise EmptyTranscriptionError("Speech-to-Text yielded an empty transcription.")

        # 5. MAI Brain execution (Authoritative P0-P11 pipeline reuse)
        mai_start = time.perf_counter()
        text_response: str = ""

        if self.chat_service:
            # Reuses complete DB persistence, conversation history, memory, active task state, and planning
            text_response = self.chat_service.chat(
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                user_message=stt_result.transcript,
            )
        elif self.orchestrator:
            # Direct orchestrator handle (for lightweight / direct unit test executions)
            text_response = self.orchestrator.handle(
                user_message=stt_result.transcript,
                conversation_messages=[],
                memory_context=[],
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            )
        else:
            raise RuntimeError("VoicePipeline requires either chat_service or orchestrator to execute.")

        mai_latency_ms = round((time.perf_counter() - mai_start) * 1000.0, 2)

        # 6. TTS Synthesis
        # Grounding invariant: TTS receives the final grounded text response, NEVER raw web/evidence text
        tts_start = time.perf_counter()
        audio_out_b64: str | None = None
        tts_error_message: str | None = None
        status = "success"

        try:
            formatted_speech = self.formatter.format_for_speech(text_response)
            tts_res = await self.tts_provider.synthesize(
                text=formatted_speech,
                voice=self.config.tts_voice,
                language=stt_result.language or self.config.default_language,
            )
            audio_out_b64 = base64.b64encode(tts_res.audio_bytes).decode("ascii")
        except (TTSError, Exception) as tts_err:
            # Preserve grounded text response; TTS failure does not fail MAI or corrupt task state
            tts_error_message = str(tts_err)
            status = "partial_success"

        tts_latency_ms = round((time.perf_counter() - tts_start) * 1000.0, 2)
        total_latency_ms = round((time.perf_counter() - total_start) * 1000.0, 2)

        from app.core.runtime_mode import determine_runtime_mode
        stt_raw = getattr(self.stt_provider, "provider_name", "assemblyai")
        stt_prov_name = stt_raw if isinstance(stt_raw, str) else str(stt_raw)
        stt_mode_raw = getattr(
            self.stt_provider,
            "stt_mode",
            "CLOUD" if stt_prov_name in ("assemblyai", "whisper", "groq", "openai") else "LOCAL_FALLBACK",
        )
        stt_mode = stt_mode_raw if isinstance(stt_mode_raw, str) else "CLOUD"

        orch = getattr(self.chat_service, "orchestrator", self.orchestrator)
        llm_obj = getattr(orch, "llm", None)
        llm_raw = getattr(llm_obj, "provider_name", "groq")
        llm_prov_name = llm_raw if isinstance(llm_raw, str) else str(llm_raw)
        llm_mode_raw = getattr(
            llm_obj,
            "runtime_mode",
            "CLOUD" if getattr(llm_obj, "provider_type", "cloud") == "cloud" else "LOCAL_FALLBACK",
        )
        llm_mode = llm_mode_raw if isinstance(llm_mode_raw, str) else "CLOUD"
        mode = determine_runtime_mode(stt_mode, llm_mode).value


        timing = VoiceTimingMetadata(
            stt_latency_ms=stt_latency_ms,
            mai_latency_ms=mai_latency_ms,
            tts_latency_ms=tts_latency_ms,
            total_latency_ms=total_latency_ms,
            stt_provider=stt_prov_name,
            llm_provider=llm_prov_name,
            runtime_mode=mode,
        )

        return VoiceChatResponse(
            conversation_id=request.conversation_id,
            transcript=stt_result.transcript,
            text_response=text_response,
            audio_base64=audio_out_b64,
            audio_content_type=self.config.tts_audio_format if audio_out_b64 else None,
            status=status,
            timing=timing,
            tts_error=tts_error_message,
        )
