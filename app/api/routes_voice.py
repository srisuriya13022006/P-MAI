"""
P12 — Voice API Endpoints.
Exposes POST /voice/chat for audio turnaround without exposing internal engine internals.
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
import asyncio
import logging
import os
import time
from sqlalchemy.orm import Session
from typing import Any

from app.database.session import get_db
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.ollama_provider import OllamaProvider, get_ollama_provider
from app.services.chat_service import ChatService
from app.voice.events import RealtimeClientEvent, RealtimeEventType, RealtimeServerEvent
from app.voice.exceptions import (
    AudioTooLargeError,
    AudioTooLongError,
    CorruptedAudioError,
    EmptyAudioError,
    EmptyTranscriptionError,
    InvalidAudioError,
    SessionExpiredError,
    SessionInactivityTimeoutError,
    STTTimeoutError,
    STTUnavailableError,
    UnsupportedAudioFormatError,
    VoiceDisabledError,
    VoiceError,
)

logger = logging.getLogger("app.voice.routes")
from app.voice.pipeline import VoicePipeline
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import VoiceChatRequest, VoiceChatResponse

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)


def get_active_voice_llm(
    canonical_llm: LLMProvider = Depends(get_llm_provider),
    legacy_llm: Any = Depends(get_ollama_provider),
) -> LLMProvider:
    """Dependency resolver supporting cloud-first primary while honoring legacy mock overrides."""
    if not isinstance(legacy_llm, OllamaProvider):
        return legacy_llm
    return canonical_llm


@router.get("/ui", response_class=FileResponse)
async def get_voice_ui():
    """Serve P-MAI desktop/web voice interface."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "client", "web", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Voice UI not found")


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    request: VoiceChatRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_active_voice_llm),
) -> VoiceChatResponse:
    """
    Voice Chat endpoint.
    Transcribes incoming audio, executes authoritative MAI reasoning, synthesizes response.
    Never exposes internal agent/planner traces.
    """
    chat_service = ChatService(db, llm=llm)
    pipeline = VoicePipeline(chat_service=chat_service)

    try:
        response = await pipeline.process_voice(request)
        return response
    except VoiceDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except UnsupportedAudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except AudioTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except (AudioTooLongError, EmptyAudioError, CorruptedAudioError, InvalidAudioError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except STTUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except STTTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except EmptyTranscriptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": exc.error_code, "message": exc.message},
        )
    except VoiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": exc.error_code, "message": exc.message},
        )


@router.websocket("/realtime")
async def websocket_realtime_voice(
    websocket: WebSocket,
    conversation_id: str = "default",
    user_id: str = "default",
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_active_voice_llm),
):
    """
    WebSocket endpoint for full-duplex real-time voice interaction.
    Accepts RealtimeClientEvent JSON messages and emits RealtimeServerEvent streams.
    Supports streaming STT, grounded MAI execution, streaming TTS, and barge-in.
    """
    await websocket.accept()

    chat_service = ChatService(db, llm=llm)
    pipeline = RealtimeVoicePipeline(chat_service=chat_service)

    session = None
    try:
        session = pipeline.session_manager.create_session(
            conversation_id=conversation_id,
            user_id=user_id,
        )
        logger.info(
            "WebSocket session created: session_id=%s, conversation_id=%s, user_id=%s",
            session.session_id, conversation_id, user_id,
        )

        while True:
            try:
                # Proactively enforce idle inactivity timeout and session lifetime bounds
                now = time.time()
                idle = now - session.last_activity_at
                remaining = session.config.realtime_inactivity_timeout_seconds - idle
                if remaining <= 0:
                    session.check_bounds()

                wait_slice = max(0.2, min(remaining, 5.0))
                text_data = await asyncio.wait_for(websocket.receive_text(), timeout=wait_slice)
            except asyncio.TimeoutError:
                session.check_bounds()
                continue
            except WebSocketDisconnect:
                raise

            try:
                client_event = RealtimeClientEvent.model_validate_json(text_data)
            except Exception as parse_err:
                err_event = RealtimeServerEvent(
                    type=RealtimeEventType.ERROR,
                    session_id=session.session_id if session else "unknown",
                    error_code="MALFORMED_REALTIME_EVENT",
                    error_message=f"Could not parse client event JSON: {parse_err}",
                )
                await websocket.send_text(err_event.model_dump_json())
                continue

            if client_event.type != RealtimeEventType.AUDIO_CHUNK:
                logger.info(
                    "WebSocket client event: session_id=%s, turn=%s, event_type=%s, session_status=%s",
                    session.session_id, session.current_turn, client_event.type.value, session.status.value,
                )

            async for server_event in pipeline.handle_client_event(session, client_event):
                await websocket.send_text(server_event.model_dump_json())
                if server_event.type in (
                    RealtimeEventType.SPEECH_START,
                    RealtimeEventType.SPEECH_END,
                    RealtimeEventType.RESPONSE_START,
                    RealtimeEventType.RESPONSE_END,
                    RealtimeEventType.ERROR,
                    RealtimeEventType.CLOSE,
                ):
                    logger.info(
                        "WebSocket server event: session_id=%s, turn=%s, event_type=%s, session_status=%s",
                        session.session_id, session.current_turn, server_event.type.value, session.status.value,
                    )
                if server_event.type == RealtimeEventType.CLOSE:
                    logger.info(
                        "WebSocket closing per CLOSE event: session_id=%s, turn=%s",
                        session.session_id, session.current_turn,
                    )
                    await websocket.close(code=1000)
                    return

    except (SessionInactivityTimeoutError, SessionExpiredError) as timeout_exc:
        sess_id = session.session_id if session else "unknown"
        turn_idx = session.current_turn if session else 0
        last_status = session.status.value if session else "CLOSED"
        err_code = "SESSION_INACTIVITY_TIMEOUT" if isinstance(timeout_exc, SessionInactivityTimeoutError) else "SESSION_EXPIRED"
        logger.warning(
            "WebSocket session boundary reached: session_id=%s, conversation_id=%s, turn=%s, last_state=%s, error=%s",
            sess_id, conversation_id, turn_idx, last_status, err_code,
        )
        if session:
            pipeline.session_manager.close_session(session.session_id)
        try:
            err_event = RealtimeServerEvent(
                type=RealtimeEventType.ERROR,
                session_id=sess_id,
                turn_index=turn_idx,
                error_code=err_code,
                error_message=str(timeout_exc),
            )
            await websocket.send_text(err_event.model_dump_json())
            close_event = RealtimeServerEvent(
                type=RealtimeEventType.CLOSE,
                session_id=sess_id,
                turn_index=turn_idx,
                metadata={"reason": err_code.lower()},
            )
            await websocket.send_text(close_event.model_dump_json())
            await websocket.close(code=1000)
        except Exception:
            pass

    except WebSocketDisconnect as disc:
        sess_id = session.session_id if session else "unknown"
        turn_idx = session.current_turn if session else 0
        last_status = session.status.value if session else "CLOSED"
        close_code = getattr(disc, "code", 1000)
        close_reason = getattr(disc, "reason", None) or "client_disconnect"
        logger.info(
            "WebSocket client disconnected: session_id=%s, conversation_id=%s, turn=%s, last_state=%s, close_code=%s, reason=%s",
            sess_id, conversation_id, turn_idx, last_status, close_code, close_reason,
        )
        if session:
            pipeline.session_manager.close_session(session.session_id)

    except Exception as exc:
        sess_id = session.session_id if session else "unknown"
        turn_idx = session.current_turn if session else 0
        last_status = session.status.value if session else "CLOSED"
        logger.error(
            "WebSocket unexpected exception: session_id=%s, conversation_id=%s, turn=%s, last_state=%s, error_type=%s, error_msg=%s",
            sess_id, conversation_id, turn_idx, last_status, type(exc).__name__, exc,
            exc_info=True,
        )
        if session:
            pipeline.session_manager.close_session(session.session_id)
        try:
            err_event = RealtimeServerEvent(
                type=RealtimeEventType.ERROR,
                session_id=sess_id,
                turn_index=turn_idx,
                error_code="INTERNAL_REALTIME_ERROR",
                error_message=str(exc),
            )
            await websocket.send_text(err_event.model_dump_json())
            close_event = RealtimeServerEvent(
                type=RealtimeEventType.CLOSE,
                session_id=sess_id,
                turn_index=turn_idx,
                metadata={"reason": "internal_error"},
            )
            await websocket.send_text(close_event.model_dump_json())
            await websocket.close(code=1011)
        except Exception:
            pass
