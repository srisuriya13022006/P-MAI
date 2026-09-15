from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:

    chat_service = ChatService(db)

    response = chat_service.chat(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        user_message=request.message,
    )

    return ChatResponse(
        conversation_id=request.conversation_id,
        response=response,
    )