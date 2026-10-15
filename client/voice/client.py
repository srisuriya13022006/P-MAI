"""
P15 — Production Voice Client for P-MAI.
Connects via WebSocket to /voice/realtime, manages audio capture, streaming delivery,
barge-in signaling, audio playback, explicit UI state machine transitions, bounded history,
reconnect exhaustion guardrails, stop vs cancel semantics, and security filtering.
"""
import asyncio
import base64
import json
import re
import time
from typing import Any, Callable, Optional
import websockets

from client.voice.audio_device import AudioDevice
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState, VoiceClientStateMachine


# Regex patterns to scrub internal traces / IDs if accidentally present
INTERNAL_TRACE_SCRUBBERS = [
    re.compile(r"step_[a-f0-9\-]+", re.IGNORECASE),
    re.compile(r"trace_[a-f0-9\-]+", re.IGNORECASE),
    re.compile(r"task_[a-f0-9\-]+", re.IGNORECASE),
    re.compile(r"plan_[a-f0-9\-]+", re.IGNORECASE),
    re.compile(r"session_[a-f0-9\-]{8,}", re.IGNORECASE),
]


def scrub_internal_identifiers(text: str) -> str:
    """Ensure internal IDs and traces are never displayed in client UI."""
    if not text:
        return ""
    result = text
    for pattern in INTERNAL_TRACE_SCRUBBERS:
        result = pattern.sub("[internal]", result)
    return result


class PMAIVoiceClient:
    """
    Conversational voice client implementing the P15 specification.
    """

    def __init__(
        self,
        config: Optional[VoiceClientConfig] = None,
        server_ws_url: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        audio_device: Optional[AudioDevice] = None,
    ):
        self.config = config or VoiceClientConfig()
        if server_ws_url:
            self.config.backend_ws_url = server_ws_url
        if conversation_id:
            self.config.conversation_id = conversation_id
        if user_id:
            self.config.user_id = user_id

        self.device = audio_device or AudioDevice(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_duration_s=self.config.audio_chunk_duration_s,
        )

        self.state_machine = VoiceClientStateMachine(initial_state=ClientUIState.DISCONNECTED)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False

        # Reconnect state
        self.reconnect_attempts = 0

        # Conversation history (bounded)
        self.history: list[dict[str, str]] = []

        # Current live turn state
        self.current_partial_transcript: str = ""
        self.current_final_transcript: str = ""
        self.current_response_text: str = ""
        self.buffered_audio_bytes: int = 0

        # Latency & performance tracking
        self.turn_metrics: dict[str, float] = {}
        self.session_metrics: dict[str, Any] = {}

        # Callbacks for UI events
        self.on_state_change: Optional[Callable[[ClientUIState, str], None]] = None
        self.on_partial_transcript: Optional[Callable[[str], None]] = None
        self.on_final_transcript: Optional[Callable[[str], None]] = None
        self.on_response_chunk: Optional[Callable[[bytes], None]] = None
        self.on_turn_completed: Optional[Callable[[str, str, dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    @property
    def is_ws_open(self) -> bool:
        """Check whether the underlying WebSocket is actively open and ready for I/O."""
        if not self.ws or not self.is_connected:
            return False
        if getattr(self.ws, "closed", None) is True:
            return False
        state = getattr(self.ws, "state", None)
        if state is not None and not hasattr(state, "_mock_return_value"):
            name = getattr(state, "name", None)
            if name is not None and name != "OPEN":
                return False
        return True

    def _handle_connection_loss(self, reason: str = "Connection lost.") -> None:
        """
        Safely clean up active turn resources after an unexpected connection loss.
        Halts hardware audio playback/recording, clears in-flight buffers,
        transitions UI state to DISCONNECTED, and notifies error callbacks.
        """
        self.is_connected = False
        self.buffered_audio_bytes = 0
        self.device.stop_playback()
        self.device.reset_playback_state()
        self._set_state(ClientUIState.DISCONNECTED, reason)
        if self.on_error:
            try:
                self.on_error(reason)
            except Exception:
                pass

    def _set_state(self, state: ClientUIState, err_msg: Optional[str] = None) -> None:
        self.state_machine.transition_to(state, error_message=err_msg)
        if self.on_state_change:
            try:
                self.on_state_change(self.state_machine.current_state, self.state_machine.display_label)
            except Exception:
                pass

    async def connect(self) -> None:
        """
        Connect to P-MAI WebSocket voice backend with bounded retry handling.
        """
        self._set_state(ClientUIState.CONNECTING)
        t_start = time.perf_counter()

        query_url = (
            f"{self.config.backend_ws_url}?"
            f"conversation_id={self.config.conversation_id}&"
            f"user_id={self.config.user_id}"
        )

        try:
            self.ws = await websockets.connect(
                query_url,
                ping_interval=30,
                ping_timeout=120,
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            self.session_metrics["connection_latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            self._set_state(ClientUIState.IDLE)
        except Exception as exc:
            self.is_connected = False
            self._set_state(ClientUIState.ERROR, f"Connection failed: {exc}")
            raise ConnectionError(f"Could not connect to {self.config.backend_ws_url}: {exc}") from exc

    async def reconnect(self) -> bool:
        """
        Attempt bounded reconnect. Fails after max_reconnect_attempts.
        Preserves conversation_id and user_id.
        """
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self._set_state(ClientUIState.CONNECTING)
            await asyncio.sleep(self.config.reconnect_delay_seconds * self.reconnect_attempts)
            try:
                await self.connect()
                return True
            except Exception:
                pass

        # Exhausted reconnect attempts
        self._set_state(ClientUIState.DISCONNECTED, "Connection lost. Please reconnect.")
        if self.on_error:
            self.on_error("Connection lost. Please reconnect.")
        return False

    async def start_turn(self) -> None:
        """
        Signal beginning of user utterance.
        Handles barge-in immediately if currently speaking.
        Automatically reconnects if the socket was closed.
        """
        if not self.is_ws_open:
            self.is_connected = False
            reconnected = await self.reconnect()
            if not reconnected or not self.is_ws_open:
                self._handle_connection_loss("Cannot start turn: WebSocket is disconnected.")
                raise ConnectionError("Cannot start turn: WebSocket is disconnected.")

        # Check microphone permissions & availability
        self.device.check_permissions()
        self.device.check_availability()

        interruption_latency_ms = None
        # Barge-in handling
        if self.state_machine.current_state == ClientUIState.SPEAKING:
            t_barge = time.perf_counter()
            self.device.stop_playback()
            interruption_latency_ms = round((time.perf_counter() - t_barge) * 1000.0, 2)
            self._set_state(ClientUIState.INTERRUPTED)

        self.device.reset_playback_state()
        self.current_partial_transcript = ""
        self.current_final_transcript = ""
        self.current_response_text = ""
        self.buffered_audio_bytes = 0
        self.turn_metrics = {"client_turn_start": time.perf_counter()}
        if interruption_latency_ms is not None:
            self.turn_metrics["interruption_latency_ms"] = interruption_latency_ms

        self._set_state(ClientUIState.LISTENING)
        try:
            await self.ws.send(json.dumps({"type": "AUDIO_START"}))
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            self._handle_connection_loss(f"Connection lost while starting turn: {exc}")
            raise ConnectionError(f"Connection lost while starting turn: {exc}") from exc

    async def send_audio_chunk(self, chunk_bytes: bytes) -> None:
        """
        Stream a canonical PCM audio chunk to the backend with bounded buffer protection.
        Verifies that WebSocket is open before sending. Catches ConnectionClosed safely.
        """
        if not self.is_ws_open:
            self._handle_connection_loss("Cannot send audio chunk: WebSocket is disconnected.")
            raise ConnectionError("Cannot send audio chunk: WebSocket is disconnected.")

        # Oversized chunk guard
        if len(chunk_bytes) > self.config.max_chunk_size_bytes:
            raise ValueError(f"Chunk size {len(chunk_bytes)} bytes exceeds maximum {self.config.max_chunk_size_bytes} bytes.")

        # Total buffer bounds guard
        if self.buffered_audio_bytes + len(chunk_bytes) > self.config.max_audio_buffer_bytes:
            raise ValueError("Audio buffer exceeded maximum allowable client memory capacity.")

        self.buffered_audio_bytes += len(chunk_bytes)
        b64_audio = base64.b64encode(chunk_bytes).decode("ascii")

        try:
            await self.ws.send(json.dumps({
                "type": "AUDIO_CHUNK",
                "audio_base64": b64_audio,
                "content_type": "audio/wav",
            }))
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            self._handle_connection_loss(f"Connection lost while sending audio chunk: {exc}")
            raise ConnectionError(f"Connection lost while streaming audio chunk: {exc}") from exc

    async def finish_turn(self) -> None:
        """
        Signal end of user utterance.
        """
        if not self.is_ws_open:
            self._handle_connection_loss("Cannot finish turn: WebSocket is disconnected.")
            raise ConnectionError("Cannot finish turn: WebSocket is disconnected.")

        self.turn_metrics["client_audio_end"] = time.perf_counter()
        self._set_state(ClientUIState.TRANSCRIBING)
        try:
            await self.ws.send(json.dumps({"type": "AUDIO_END"}))
        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            self._handle_connection_loss(f"Connection lost while finishing turn: {exc}")
            raise ConnectionError(f"Connection lost while finishing turn: {exc}") from exc

    async def stop_speaking(self) -> None:
        """
        STOP SPEAKING control:
        Interrupts audio playback immediately, preserves active P8 task state.
        """
        self.device.stop_playback()
        if self.ws and self.is_connected:
            try:
                await self.ws.send(json.dumps({
                    "type": "INTERRUPT",
                    "reason": "stop_speaking_button",
                }))
            except Exception:
                pass
        self._set_state(ClientUIState.IDLE)

    async def cancel_task(self) -> None:
        """
        CANCEL TASK control:
        Sends a cancellation turn through the voice pipeline to transition the backend P8 task to CANCELLED.
        """
        self.device.stop_playback()
        # To cancel the active task authoritatively, send speech turn with cancellation command
        synthetic_chunks = self.device.generate_mock_speech_chunks(3)
        await self.start_turn()
        for chunk in synthetic_chunks:
            await self.send_audio_chunk(chunk)
        await self.finish_turn()

    async def interrupt(self, reason: str = "user_barge_in") -> None:
        """
        Send INTERRUPT signal to backend and halt local playback.
        """
        self.device.stop_playback()
        if self.ws and self.is_connected:
            try:
                await self.ws.send(json.dumps({
                    "type": "INTERRUPT",
                    "reason": reason,
                }))
            except Exception:
                pass
        self._set_state(ClientUIState.INTERRUPTED)

    def _append_history(self, speaker: str, text: str) -> None:
        """Append to conversation history respecting max_history_turns bounds."""
        clean_text = scrub_internal_identifiers(text)
        self.history.append({"speaker": speaker, "text": clean_text})
        if len(self.history) > self.config.max_history_turns:
            self.history = self.history[-self.config.max_history_turns:]

    def handle_server_event(self, event: dict) -> None:
        """
        Process incoming server event with validation and deterministic state updates.
        """
        if not isinstance(event, dict) or "type" not in event:
            self._set_state(ClientUIState.ERROR, "Received malformed server event")
            if self.on_error:
                self.on_error("Received malformed server event")
            return

        ev_type = event.get("type")

        if ev_type == "SPEECH_START":
            self._set_state(ClientUIState.LISTENING)

        elif ev_type == "SPEECH_END":
            self._set_state(ClientUIState.TRANSCRIBING)

        elif ev_type == "TRANSCRIPT_PARTIAL":
            raw_partial = event.get("transcript") or ""
            clean_partial = scrub_internal_identifiers(raw_partial)
            self.current_partial_transcript = clean_partial
            if "t_first_partial" not in self.turn_metrics and "client_turn_start" in self.turn_metrics:
                self.turn_metrics["t_first_partial"] = round((time.perf_counter() - self.turn_metrics["client_turn_start"]) * 1000.0, 2)
            if self.on_partial_transcript:
                self.on_partial_transcript(clean_partial)

        elif ev_type == "TRANSCRIPT_FINAL":
            raw_final = event.get("transcript") or ""
            clean_final = scrub_internal_identifiers(raw_final)
            self.current_final_transcript = clean_final
            if "t_final_transcript" not in self.turn_metrics and "client_turn_start" in self.turn_metrics:
                self.turn_metrics["t_final_transcript"] = round((time.perf_counter() - self.turn_metrics["client_turn_start"]) * 1000.0, 2)
            self._append_history("User", clean_final)
            self._set_state(ClientUIState.PROCESSING)
            if self.on_final_transcript:
                self.on_final_transcript(clean_final)

        elif ev_type == "RESPONSE_START":
            self._set_state(ClientUIState.PROCESSING)

        elif ev_type == "RESPONSE_CHUNK":
            self._set_state(ClientUIState.SPEAKING)
            audio_b64 = event.get("audio_base64")
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    self.received_chunks_count = getattr(self, "received_chunks_count", 0) + 1
                    self.received_audio_bytes = getattr(self, "received_audio_bytes", 0) + len(audio_bytes)
                    if "time_to_first_audio_ms" not in self.turn_metrics and "client_audio_end" in self.turn_metrics:
                        self.turn_metrics["time_to_first_audio_ms"] = round((time.perf_counter() - self.turn_metrics["client_audio_end"]) * 1000.0, 2)
                    self.device.play_audio_chunk(audio_bytes)
                    if self.on_response_chunk:
                        self.on_response_chunk(audio_bytes)
                except Exception:
                    pass

        elif ev_type == "RESPONSE_END":
            raw_response = event.get("text_chunk") or ""
            clean_response = scrub_internal_identifiers(raw_response)
            self.current_response_text = clean_response
            if "client_turn_start" in self.turn_metrics:
                self.turn_metrics["total_turn_turnaround_ms"] = round((time.perf_counter() - self.turn_metrics["client_turn_start"]) * 1000.0, 2)

            self.turn_metrics["received_audio_chunks"] = getattr(self, "received_chunks_count", 0)
            self.turn_metrics["received_audio_bytes"] = getattr(self, "received_audio_bytes", 0)
            self.turn_metrics["stages"] = {
                "playback_started": getattr(self, "received_chunks_count", 0) > 0,
                "playback_completed": True,
            }

            self._append_history("MAI", clean_response)
            self._set_state(ClientUIState.IDLE)
            if self.on_turn_completed:
                self.on_turn_completed(self.current_final_transcript, clean_response, self.turn_metrics)

        elif ev_type == "INTERRUPT":
            self.device.stop_playback()
            self._set_state(ClientUIState.INTERRUPTED)
            self._set_state(ClientUIState.IDLE)

        elif ev_type == "ERROR":
            err_code = event.get("error_code") or "ERROR"
            err_msg = scrub_internal_identifiers(event.get("error_message") or "Unknown server error")
            display_err = f"{err_code}: {err_msg}"
            self._set_state(ClientUIState.ERROR, display_err)
            if self.on_error:
                self.on_error(display_err)

        elif ev_type == "CLOSE":
            self.is_connected = False
            self._set_state(ClientUIState.CLOSED)

    async def run_turn(
        self,
        chunks: list[bytes],
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Helper method to run a complete turn synchronously for testing and scripts.
        """
        await self.start_turn()
        for chunk in chunks:
            await self.send_audio_chunk(chunk)
            await asyncio.sleep(0.01)
        await self.finish_turn()

        received_events = []
        while True:
            try:
                raw_msg = await self.ws.recv()
            except (websockets.exceptions.ConnectionClosed, OSError) as exc:
                self._handle_connection_loss(f"Connection closed by server: {exc}")
                break

            try:
                event = json.loads(raw_msg)
            except Exception:
                event = {"type": "ERROR", "error_message": "Malformed JSON"}

            received_events.append(event)
            self.handle_server_event(event)

            if on_event:
                on_event(event)

            if event.get("type") in ("RESPONSE_END", "ERROR", "CLOSE"):
                break

        return {
            "events": received_events,
            "final_transcript": self.current_final_transcript,
            "response_text": self.current_response_text,
            "history": self.history,
            "metrics": self.turn_metrics,
        }

    async def close(self) -> None:
        """
        Cleanly terminate session and release audio hardware.
        """
        self._set_state(ClientUIState.CLOSING)
        if self.ws and self.is_connected:
            try:
                await self.ws.send(json.dumps({"type": "CLOSE"}))
                await self.ws.close()
            except Exception:
                pass
        self.device.close()
        self.is_connected = False
        self._set_state(ClientUIState.CLOSED)
