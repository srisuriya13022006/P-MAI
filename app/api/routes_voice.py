"""
P12 — Voice API Endpoints.
Exposes POST /voice/chat for audio turnaround without exposing internal engine internals.
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
import os
from sqlalchemy.orm import Session

from app.database.session import get_db
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
    STTTimeoutError,
    STTUnavailableError,
    UnsupportedAudioFormatError,
    VoiceDisabledError,
    VoiceError,
)
from app.voice.pipeline import VoicePipeline
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import VoiceChatRequest, VoiceChatResponse

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)


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
    llm: OllamaProvider = Depends(get_ollama_provider),
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
    llm: OllamaProvider = Depends(get_ollama_provider),
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

        while True:
            text_data = await websocket.receive_text()
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

            async for server_event in pipeline.handle_client_event(session, client_event):
                await websocket.send_text(server_event.model_dump_json())
                if server_event.type == RealtimeEventType.CLOSE:
                    await websocket.close()
                    return

    except WebSocketDisconnect:
        if session:
            pipeline.session_manager.close_session(session.session_id)
    except Exception as exc:
        if session:
            pipeline.session_manager.close_session(session.session_id)
        try:
            err_event = RealtimeServerEvent(
                type=RealtimeEventType.ERROR,
                session_id=session.session_id if session else "unknown",
                error_code="INTERNAL_REALTIME_ERROR",
                error_message=str(exc),
            )
            await websocket.send_text(err_event.model_dump_json())
            await websocket.close()
        except Exception:
            pass
