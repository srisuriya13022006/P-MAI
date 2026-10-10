"""
P12.2 — Real-time Voice Interaction Pipeline.
Coordinates streaming audio, VAD, streaming STT, MAI Brain, streaming TTS, and barge-in.
Enforces that voice remains strictly an I/O presentation layer.
"""
import asyncio
import base64
import binascii
import re
import time
from typing import AsyncGenerator

from app.agent.orchestrator import MAIOrchestrator
from app.services.chat_service import ChatService
from app.voice.config import VoiceConfig
from app.voice.events import (
    RealtimeClientEvent,
    RealtimeEventType,
    RealtimeServerEvent,
)
from app.voice.exceptions import (
    BackpressureError,
    EmptyTranscriptionError,
    InvalidSessionStateTransitionError,
    MalformedRealtimeEventError,
    RealtimeVoiceError,
    STTError,
    TTSError,
    VoiceDisabledError,
)
from app.voice.schemas import STTResult, TTSResult
from app.voice.session import (
    RealtimeVoiceSession,
    SessionStatus,
    VoiceSessionManager,
)
from app.voice.audio import AudioNormalizer
from app.voice.formatter import VoicePresentationFormatter
from app.voice.stt.base import SpeechToTextProvider
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.mock import MockSTTProvider
from app.voice.stt.streaming import (
    MockStreamingSTTProvider,
    StreamingSpeechToTextProvider,
)
from app.voice.tts.base import TextToSpeechProvider
from app.voice.tts.factory import create_tts_provider
from app.voice.tts.mock import MockTTSProvider
from app.voice.tts.streaming import (
    MockStreamingTTSProvider,
    StreamingTextToSpeechProvider,
)
from app.voice.vad.base import VoiceActivityDetector
from app.voice.vad.detector import EnergyVADDetector


class RealtimeVoicePipeline:
    """
    Coordinates full-duplex real-time conversational voice interaction.
    """

    def __init__(
        self,
        chat_service: ChatService | None = None,
        orchestrator: MAIOrchestrator | None = None,
        stt_provider: StreamingSpeechToTextProvider | SpeechToTextProvider | None = None,
        tts_provider: StreamingTextToSpeechProvider | TextToSpeechProvider | None = None,
        vad_detector: VoiceActivityDetector | None = None,
        session_manager: VoiceSessionManager | None = None,
        config: VoiceConfig | None = None,
    ):
        self.config = config or VoiceConfig.from_settings()
        self.chat_service = chat_service
        self.orchestrator = orchestrator

        self.session_manager = session_manager or VoiceSessionManager(self.config)
        self.formatter = VoicePresentationFormatter()
        self.normalizer = AudioNormalizer()

        self.stt_provider = stt_provider or self._resolve_stt_provider()
        self.tts_provider = tts_provider or self._resolve_tts_provider()
        self.vad_detector = vad_detector or EnergyVADDetector(
            silence_threshold_db=self.config.vad_silence_threshold_db,
            max_duration_seconds=self.config.realtime_max_utterance_duration_seconds,
        )

    def _resolve_stt_provider(self) -> SpeechToTextProvider:
        provider_name = (self.config.stt_provider or "").lower().strip()
        if provider_name == "mock":
            return MockStreamingSTTProvider()
        return create_stt_provider(self.config)

    def _resolve_tts_provider(self) -> TextToSpeechProvider:
        provider_name = (self.config.tts_provider or "").lower().strip()
        if provider_name == "mock":
            return MockStreamingTTSProvider()
        return create_tts_provider(self.config)

    def is_stop_speaking_command(self, text: str) -> bool:
        """
        Check if text is purely an instruction to stop talking (barge-in / mute).
        Must NOT be confused with 'cancel this task'.
        """
        cleaned = text.strip().lower().rstrip(".,!?")
        # Direct check against stop phrases
        for stop_p in self.config.realtime_stop_phrases:
            if cleaned == stop_p:
                return True
        return False

    async def handle_client_event(
        self,
        session: RealtimeVoiceSession,
        event: RealtimeClientEvent,
    ) -> AsyncGenerator[RealtimeServerEvent, None]:
        """
        Process incoming streaming client event and yield outgoing server events.
        """
        if not self.config.enabled or not self.config.realtime_enabled:
            raise VoiceDisabledError("Real-time voice interaction is disabled.")

        session.check_bounds()
        session.touch()

        # -------------------------------------------------------------
        # 1. AUDIO_START
        # -------------------------------------------------------------
        if event.type == RealtimeEventType.AUDIO_START:
            # Check for barge-in if currently speaking
            if session.status == SessionStatus.SPEAKING:
                session.is_interrupted = True
                session.interruption_count += 1
                session.transition_to(SessionStatus.INTERRUPTED)
                yield RealtimeServerEvent(
                    type=RealtimeEventType.INTERRUPT,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    is_interrupted=True,
                    metadata={"reason": "barge_in_on_audio_start"},
                )
                session.transition_to(SessionStatus.LISTENING)
            elif session.status in (SessionStatus.IDLE, SessionStatus.INTERRUPTED):
                session.transition_to(SessionStatus.LISTENING)

            session.is_speech_active = True
            session.is_interrupted = False
            session.clear_audio_buffer()

            if hasattr(self.stt_provider, "reset"):
                self.stt_provider.reset()

            yield RealtimeServerEvent(
                type=RealtimeEventType.SPEECH_START,
                session_id=session.session_id,
                turn_index=session.current_turn,
            )

        # -------------------------------------------------------------
        # 2. AUDIO_CHUNK
        # -------------------------------------------------------------
        elif event.type == RealtimeEventType.AUDIO_CHUNK:
            if not event.audio_base64:
                return

            try:
                chunk_bytes = base64.b64decode(event.audio_base64)
            except (binascii.Error, ValueError) as err:
                raise MalformedRealtimeEventError(f"Malformed audio chunk base64: {err}")

            # Barge-in detection during speaking
            if session.status == SessionStatus.SPEAKING:
                session.is_interrupted = True
                session.interruption_count += 1
                session.transition_to(SessionStatus.INTERRUPTED)
                yield RealtimeServerEvent(
                    type=RealtimeEventType.INTERRUPT,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    is_interrupted=True,
                    metadata={"reason": "barge_in_on_audio_chunk"},
                )
                session.transition_to(SessionStatus.LISTENING)

            # Append chunk with backpressure limit
            session.append_audio_chunk(chunk_bytes)

            # Feed chunk to streaming STT if supported
            if isinstance(self.stt_provider, StreamingSpeechToTextProvider):
                try:
                    partials = await self.stt_provider.feed_audio_chunk(
                        chunk_bytes,
                        content_type=event.content_type or "audio/wav",
                    )
                    for part in partials:
                        if part.transcript:
                            session.current_stt_transcript = part.transcript
                            yield RealtimeServerEvent(
                                type=RealtimeEventType.TRANSCRIPT_PARTIAL,
                                session_id=session.session_id,
                                turn_index=session.current_turn,
                                transcript=part.transcript,
                            )
                except (STTError, Exception) as exc:
                    yield RealtimeServerEvent(
                        type=RealtimeEventType.ERROR,
                        session_id=session.session_id,
                        turn_index=session.current_turn,
                        error_code="STT_STREAMING_ERROR",
                        error_message=str(exc),
                    )

        # -------------------------------------------------------------
        # 3. AUDIO_END
        # -------------------------------------------------------------
        elif event.type == RealtimeEventType.AUDIO_END:
            t_server_recv_audio_end = time.perf_counter()
            if session.status == SessionStatus.LISTENING:
                session.transition_to(SessionStatus.TRANSCRIBING)

            yield RealtimeServerEvent(
                type=RealtimeEventType.SPEECH_END,
                session_id=session.session_id,
                turn_index=session.current_turn,
            )

            # Finalize Speech-to-Text
            final_transcript = ""
            stt_latency_ms = 0.0
            stt_res = None
            t_stt_start = time.perf_counter()
            try:
                if isinstance(self.stt_provider, StreamingSpeechToTextProvider):
                    stt_res = await self.stt_provider.finalize_stream()
                    final_transcript = stt_res.transcript
                else:
                    audio_data = session.clear_audio_buffer()
                    stt_res = await self.stt_provider.transcribe(
                        audio_data,
                        content_type=event.content_type or "audio/wav",
                    )
                    final_transcript = stt_res.transcript
            except (STTError, Exception) as stt_err:
                yield RealtimeServerEvent(
                    type=RealtimeEventType.ERROR,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    error_code="STT_FINALIZATION_ERROR",
                    error_message=str(stt_err),
                )
                session.transition_to(SessionStatus.LISTENING)
                return

            t_stt_end = time.perf_counter()
            stt_latency_ms = round((t_stt_end - t_stt_start) * 1000.0, 2)

            if not final_transcript or not final_transcript.strip():
                yield RealtimeServerEvent(
                    type=RealtimeEventType.ERROR,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    error_code="EMPTY_TRANSCRIPTION",
                    error_message="Final transcript was empty.",
                )
                session.transition_to(SessionStatus.LISTENING)
                return

            session.current_stt_transcript = final_transcript.strip()
            yield RealtimeServerEvent(
                type=RealtimeEventType.TRANSCRIPT_FINAL,
                session_id=session.session_id,
                turn_index=session.current_turn,
                transcript=final_transcript.strip(),
            )

            # Check for pure "Stop" / "Stop talking" semantics
            if self.is_stop_speaking_command(final_transcript):
                # Stop talking only; do NOT cancel tasks in P8
                yield RealtimeServerEvent(
                    type=RealtimeEventType.RESPONSE_END,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    text_chunk="Playback stopped.",
                    metadata={"action": "stop_speaking_only"},
                )
                session.transition_to(SessionStatus.LISTENING)
                return

            # Invoke authoritative MAI Brain
            session.transition_to(SessionStatus.PROCESSING)
            session.current_turn += 1

            yield RealtimeServerEvent(
                type=RealtimeEventType.RESPONSE_START,
                session_id=session.session_id,
                turn_index=session.current_turn,
            )

            mai_start = time.perf_counter()
            grounded_response: str = ""

            try:
                if self.chat_service:
                    grounded_response = await asyncio.to_thread(
                        self.chat_service.chat,
                        conversation_id=session.conversation_id,
                        user_id=session.user_id,
                        user_message=final_transcript.strip(),
                    )
                elif self.orchestrator:
                    grounded_response = await asyncio.to_thread(
                        self.orchestrator.handle,
                        user_message=final_transcript.strip(),
                        conversation_messages=[],
                        memory_context=[],
                        user_id=session.user_id,
                        conversation_id=session.conversation_id,
                    )
                else:
                    raise RuntimeError("RealtimeVoicePipeline requires chat_service or orchestrator.")
            except Exception as mai_err:
                yield RealtimeServerEvent(
                    type=RealtimeEventType.ERROR,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    error_code="MAI_PROCESSING_ERROR",
                    error_message=str(mai_err),
                )
                session.transition_to(SessionStatus.LISTENING)
                return

            mai_end = time.perf_counter()
            mai_latency_ms = round((mai_end - mai_start) * 1000.0, 2)
            session.current_response_text = grounded_response

            # Stream TTS Output
            session.transition_to(SessionStatus.SPEAKING)
            tts_start = time.perf_counter()
            t_first_tts_chunk = None
            chunk_index = 0

            try:
                formatted_speech = self.formatter.format_for_speech(grounded_response)
                if len(formatted_speech) > 450:
                    sentences = re.split(r"(?<=[.!?])\s+", formatted_speech)
                    shortened = []
                    curr_len = 0
                    for s in sentences:
                        shortened.append(s)
                        curr_len += len(s)
                        if curr_len >= 300:
                            break
                    speech_to_synthesize = " ".join(shortened)
                else:
                    speech_to_synthesize = formatted_speech

                if isinstance(self.tts_provider, StreamingTextToSpeechProvider):
                    async for tts_chunk in self.tts_provider.synthesize_stream(
                        text=speech_to_synthesize,
                        voice=self.config.default_voice,
                        language=self.config.default_language,
                        chunk_duration_seconds=self.config.realtime_tts_chunk_duration_seconds,
                    ):
                        # If barge-in occurred while yielding chunks, abort playback immediately
                        if session.is_interrupted or session.status == SessionStatus.INTERRUPTED:
                            break

                        if t_first_tts_chunk is None:
                            t_first_tts_chunk = time.perf_counter()

                        chunk_index += 1
                        yield RealtimeServerEvent(
                            type=RealtimeEventType.RESPONSE_CHUNK,
                            session_id=session.session_id,
                            turn_index=session.current_turn,
                            audio_base64=base64.b64encode(tts_chunk.audio_bytes).decode("ascii"),
                            metadata={"chunk_index": chunk_index},
                        )
                else:
                    tts_res = await self.tts_provider.synthesize(
                        text=formatted_speech,
                        voice=self.config.default_voice,
                        language=self.config.default_language,
                    )
                    t_first_tts_chunk = time.perf_counter()
                    if not session.is_interrupted and session.status == SessionStatus.SPEAKING:
                        yield RealtimeServerEvent(
                            type=RealtimeEventType.RESPONSE_CHUNK,
                            session_id=session.session_id,
                            turn_index=session.current_turn,
                            audio_base64=base64.b64encode(tts_res.audio_bytes).decode("ascii"),
                            metadata={"chunk_index": 1},
                        )
            except (TTSError, Exception) as tts_err:
                # TTS failure is non-fatal: preserve grounded text response
                yield RealtimeServerEvent(
                    type=RealtimeEventType.ERROR,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    error_code="TTS_STREAMING_ERROR",
                    error_message=str(tts_err),
                )

            tts_end = time.perf_counter()
            tts_latency_ms = round((tts_end - tts_start) * 1000.0, 2)

            # If not interrupted, emit RESPONSE_END with final grounded text
            if not session.is_interrupted and session.status == SessionStatus.SPEAKING:
                from app.core.runtime_mode import determine_runtime_mode
                stt_prov = getattr(self.stt_provider, "provider_name", "assemblyai")
                stt_md = getattr(self.stt_provider, "stt_mode", "CLOUD" if stt_prov in ("assemblyai", "whisper", "groq", "openai") else "LOCAL_FALLBACK")
                orch = getattr(self.chat_service, "orchestrator", self.orchestrator)
                llm_obj = getattr(orch, "llm", None)
                llm_prov = getattr(llm_obj, "provider_name", "groq")
                llm_md = getattr(llm_obj, "runtime_mode", "CLOUD" if getattr(llm_obj, "provider_type", "cloud") == "cloud" else "LOCAL_FALLBACK")
                rt_mode = determine_runtime_mode(stt_md, llm_md).value

                llm_accounting = getattr(llm_obj, "get_turn_accounting", lambda: {})() if llm_obj else {}
                stt_model_name = getattr(stt_res, "model_name", None) or getattr(self.stt_provider, "model_name", "universal-3-6-pro")

                yield RealtimeServerEvent(
                    type=RealtimeEventType.RESPONSE_END,
                    session_id=session.session_id,
                    turn_index=session.current_turn,
                    text_chunk=grounded_response,
                    is_interrupted=False,
                    metadata={
                        "stt_latency_ms": stt_latency_ms,
                        "stt_provider": stt_prov,
                        "stt_model": stt_model_name,
                        "mai_latency_ms": mai_latency_ms,
                        "tts_latency_ms": tts_latency_ms,
                        "runtime_mode": rt_mode,
                        "llm_provider": llm_prov,
                        "llm_accounting": llm_accounting,
                        "timings": {
                            "t_server_recv_audio_end": t_server_recv_audio_end,
                            "t_stt_start": t_stt_start,
                            "t_stt_end": t_stt_end,
                            "t_agent_start": mai_start,
                            "t_agent_end": mai_end,
                            "t_tts_start": tts_start,
                            "t_first_tts_chunk": t_first_tts_chunk or tts_end,
                            "t_tts_end": tts_end,
                        },
                    },
                )
                session.transition_to(SessionStatus.LISTENING)

        # -------------------------------------------------------------
        # 4. INTERRUPT
        # -------------------------------------------------------------
        elif event.type == RealtimeEventType.INTERRUPT:
            session.is_interrupted = True
            session.interruption_count += 1
            if session.status in (SessionStatus.SPEAKING, SessionStatus.PROCESSING):
                session.transition_to(SessionStatus.INTERRUPTED)
            yield RealtimeServerEvent(
                type=RealtimeEventType.INTERRUPT,
                session_id=session.session_id,
                turn_index=session.current_turn,
                is_interrupted=True,
                metadata={"reason": event.reason or "client_interrupted"},
            )
            if session.status == SessionStatus.INTERRUPTED:
                session.transition_to(SessionStatus.LISTENING)

        # -------------------------------------------------------------
        # 5. CLOSE
        # -------------------------------------------------------------
        elif event.type == RealtimeEventType.CLOSE:
            session.transition_to(SessionStatus.CLOSED)
            self.session_manager.close_session(session.session_id)
            yield RealtimeServerEvent(
                type=RealtimeEventType.CLOSE,
                session_id=session.session_id,
                turn_index=session.current_turn,
            )
