"""
P15.4 — AssemblyAI Realtime Streaming Speech-to-Text Adapter.
Provides low-latency cloud speech recognition using AssemblyAI's Universal-3 architecture.
Strictly ensures backend-only credentials and isolated transcription handling.
"""
import asyncio
import io
import logging
import os
import struct
import time
from typing import Any

import httpx

from app.voice.exceptions import (
    EmptyTranscriptionError,
    STTError,
    STTTimeoutError,
    STTUnavailableError,
)
from app.voice.schemas import STTResult
from app.voice.stt.streaming import StreamingSpeechToTextProvider

logger = logging.getLogger(__name__)


def is_wav_container(audio_bytes: bytes) -> bool:
    """Check if audio bytes contain a canonical RIFF/WAVE container header."""
    return len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def ensure_wav_container(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Ensure audio bytes are wrapped in a canonical 44-byte WAV container.
    If the bytes are raw PCM16, wraps them with valid RIFF/WAVE headers.
    If already a WAV container, returns as-is.
    """
    if is_wav_container(audio_bytes):
        return audio_bytes

    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(audio_bytes)
    total_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        total_size,
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size for PCM
        1,   # AudioFormat (1 = PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_bytes


def strip_wav_header_if_present(audio_bytes: bytes) -> bytes:
    """
    If audio_bytes starts with RIFF/WAVE container, strip header to yield raw PCM16 binary frames.
    AssemblyAI Realtime streaming WebSocket expects raw PCM16 without container headers.
    """
    if is_wav_container(audio_bytes):
        data_idx = audio_bytes.find(b"data")
        if data_idx != -1 and len(audio_bytes) >= data_idx + 8:
            return audio_bytes[data_idx + 8:]
        return audio_bytes[44:]
    return audio_bytes



class AssemblyAIRealtimeSTTAdapter(StreamingSpeechToTextProvider):
    """
    AssemblyAI Realtime Streaming and Batch Speech-to-Text Provider.
    Acts as the primary low-latency cloud STT engine for MAI.
    Uses AssemblyAI v3 Realtime Streaming WebSocket (wss://streaming.assemblyai.com/v3/ws)
    for sub-second live speech-to-text with universal-3-6-pro, with REST upload fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "universal-3-6-pro",
        timeout_seconds: float = 15.0,
        sample_rate: int = 16000,
        language: str = "en",
    ):
        self._api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.sample_rate = sample_rate
        self.language = language

        self._audio_buffer = bytearray()
        self._current_partial: str = ""
        self._final_transcript: str = ""
        self._effective_model: str | None = None
        self._is_stream_open: bool = False
        self._ws = None
        self._ws_reader_task = None
        self._stream_failed: bool = False

        if not self._api_key:
            logger.debug("AssemblyAI adapter initialized without API key (fallback will trigger if invoked).")

    @property
    def provider_name(self) -> str:
        return "assemblyai"

    def _require_api_key(self) -> str:
        key = self._api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not key:
            raise STTUnavailableError(
                "AssemblyAI API key is not configured. "
                "Set ASSEMBLYAI_API_KEY in the server environment."
            )
        return key

    async def _ensure_ws_connected(self) -> None:
        """Connect to AssemblyAI v3 Realtime WebSocket endpoint."""
        if self._ws is not None and not getattr(self._ws, "closed", False):
            return
        if self._stream_failed:
            return

        api_key = self._require_api_key()
        url = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={self.sample_rate}&speech_model={self.model_name}"
        headers = {"Authorization": api_key}
        try:
            import json
            import websockets
            self._ws = await websockets.connect(url, additional_headers=headers)
            begin_raw = await asyncio.wait_for(self._ws.recv(), timeout=3.0)
            begin_data = json.loads(begin_raw)
            self._effective_model = begin_data.get("configuration", {}).get("model", self.model_name)
            logger.info("AssemblyAI v3 streaming connected. Effective model: %s", self._effective_model)
            self._ws_reader_task = asyncio.create_task(self._read_ws_messages())
        except Exception as exc:
            logger.warning("Failed to connect to AssemblyAI v3 streaming websocket (%s); will use REST fallback", exc)
            self._stream_failed = True
            self._ws = None

    async def _read_ws_messages(self) -> None:
        """Asynchronously process Turn messages from streaming WebSocket."""
        try:
            import json
            while self._ws and not getattr(self._ws, "closed", False):
                msg = await self._ws.recv()
                data = json.loads(msg)
                mtype = data.get("type")
                logger.debug("AssemblyAI WS recv: %s", data)
                if mtype == "Turn":
                    text = data.get("transcript", "")
                    if text:
                        self._current_partial = text
                    if data.get("end_of_turn"):
                        self._final_transcript = text
                elif mtype == "Termination":
                    break
        except Exception as exc:
            logger.debug("AssemblyAI v3 WS reader ended: %s", exc)

    async def feed_audio_chunk(
        self,
        chunk: bytes,
        content_type: str = "audio/wav",
    ) -> list[STTResult]:
        """
        Ingest incremental audio chunks.
        Streams raw PCM16 frames into AssemblyAI v3 WebSocket and buffers for fallback.
        """
        self._require_api_key()
        raw_pcm = strip_wav_header_if_present(chunk)
        self._audio_buffer.extend(raw_pcm)

        await self._ensure_ws_connected()
        if self._ws and not getattr(self._ws, "closed", False):
            try:
                await self._ws.send(raw_pcm)
            except Exception as exc:
                logger.warning("Error streaming chunk to AssemblyAI websocket (%s); fallback will trigger on finalize", exc)
                self._stream_failed = True

        if self._current_partial:
            return [STTResult(transcript=self._current_partial, provider=self.provider_name, is_final=False, is_partial=True)]
        return []

    async def finalize_stream(self) -> STTResult:
        """
        Finalize stream and retrieve the recognized transcript.
        Prioritizes sub-second realtime streaming result, falling back to REST if needed.
        """
        if not self._audio_buffer:
            raise EmptyTranscriptionError("Audio buffer is empty; nothing to transcribe.")

        audio_bytes = bytes(self._audio_buffer)

        # 1. Gracefully close streaming websocket and retrieve final Turn
        if self._ws and not getattr(self._ws, "closed", False) and not self._stream_failed:
            try:
                import json
                await self._ws.send(json.dumps({"type": "Terminate"}))
                if self._ws_reader_task:
                    try:
                        await asyncio.wait_for(self._ws_reader_task, timeout=1.5)
                    except asyncio.TimeoutError:
                        pass
                await self._ws.close()
            except Exception as exc:
                logger.debug("Error finalizing AssemblyAI websocket: %s", exc)
            finally:
                self._ws = None
                self._ws_reader_task = None

        final_text = (self._final_transcript or self._current_partial).strip()
        if final_text:
            eff_model = self._effective_model or self.model_name
            self.reset()
            return STTResult(
                transcript=final_text,
                confidence=1.0,
                provider=self.provider_name,
                is_final=True,
            )

        # 2. Realtime WebSocket did not yield text: fall back to REST transcribe
        logger.info("Realtime streaming did not yield final text, using REST transcribe fallback.")
        self.reset()
        return await self.transcribe(audio_bytes)

    def reset(self) -> None:
        """Reset audio stream buffer and streaming state for next turn."""
        self._audio_buffer.clear()
        self._current_partial = ""
        self._final_transcript = ""
        self._stream_failed = False
        old_ws = self._ws
        self._ws = None
        self._ws_reader_task = None
        if old_ws and not getattr(old_ws, "closed", False):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(old_ws.close())
            except Exception:
                pass

    async def transcribe(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """
        Transcribe complete audio bytes using AssemblyAI v2 API.
        """
        api_key = self._require_api_key()
        if not audio_bytes:
            raise EmptyTranscriptionError("Audio payload is empty.")

        start_time = time.perf_counter()
        headers = {
            "Authorization": api_key,
        }

        # Guarantee canonical 44-byte WAV container for AssemblyAI v2 upload endpoint
        upload_payload = ensure_wav_container(audio_bytes, sample_rate=self.sample_rate)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # 1. Upload audio
                upload_res = await client.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    content=upload_payload,
                )
                if upload_res.status_code in (401, 403):
                    raise STTUnavailableError(
                        f"AssemblyAI authentication failed ({upload_res.status_code}): invalid or unauthorized API key."
                    )
                if upload_res.status_code == 429:
                    raise STTUnavailableError(f"AssemblyAI rate limited (429): {upload_res.text[:200]}")
                if upload_res.status_code != 200:
                    raise STTUnavailableError(
                        f"AssemblyAI upload failed with status {upload_res.status_code}: {upload_res.text[:200]}"
                    )

                upload_url = upload_res.json().get("upload_url")
                if not upload_url:
                    raise STTError("AssemblyAI did not return an upload URL.")

                # 2. Request transcription
                payload: dict[str, Any] = {
                    "audio_url": upload_url,
                    "language_code": language or self.language or "en",
                }
                if self.model_name:
                    rest_model = "universal-3-5-pro" if "3-6" in str(self.model_name) else self.model_name
                    payload["speech_models"] = [rest_model] if isinstance(rest_model, str) else list(rest_model)


                submit_res = await client.post(
                    "https://api.assemblyai.com/v2/transcript",
                    headers=headers,
                    json=payload,
                )
                if submit_res.status_code in (401, 403):
                    raise STTUnavailableError(
                        f"AssemblyAI authentication failed ({submit_res.status_code}): invalid or unauthorized API key."
                    )
                if submit_res.status_code == 429:
                    raise STTUnavailableError(f"AssemblyAI rate limited (429): {submit_res.text[:200]}")
                if submit_res.status_code != 200:
                    raise STTUnavailableError(
                        f"AssemblyAI transcript submission failed ({submit_res.status_code}): {submit_res.text[:200]}"
                    )


                transcript_id = submit_res.json().get("id")
                if not transcript_id:
                    raise STTError("AssemblyAI did not return a transcript ID.")

                # 3. Poll for completion
                poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                poll_start = time.perf_counter()

                while time.perf_counter() - poll_start < self.timeout_seconds:
                    poll_res = await client.get(poll_url, headers=headers)
                    if poll_res.status_code != 200:
                        raise STTError(f"AssemblyAI polling failed with status {poll_res.status_code}.")

                    data = poll_res.json()
                    status = data.get("status")

                    if status == "completed":
                        text = (data.get("text") or "").strip()
                        confidence = float(data.get("confidence") or 0.95)
                        detected_lang = data.get("language_code") or self.language or "en"

                        if not text:
                            raise EmptyTranscriptionError("AssemblyAI returned an empty transcription.")

                        return STTResult(
                            transcript=text,
                            language=detected_lang,
                            confidence=confidence,
                            provider=self.provider_name,
                            is_final=True,
                        )

                    elif status == "error":
                        err_msg = data.get("error", "Unknown AssemblyAI transcription error")
                        raise STTError(f"AssemblyAI processing error: {err_msg}")

                    await asyncio.sleep(0.3)

                raise STTTimeoutError(f"AssemblyAI transcription timed out after {self.timeout_seconds}s.")

        except httpx.TimeoutException as exc:
            raise STTTimeoutError(f"AssemblyAI request timed out: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise STTUnavailableError(f"AssemblyAI network connection failed: {exc}") from exc
        except (STTUnavailableError, STTTimeoutError, EmptyTranscriptionError, STTError):
            raise
        except Exception as exc:
            raise STTError(f"AssemblyAI transcription failed: {exc}") from exc
