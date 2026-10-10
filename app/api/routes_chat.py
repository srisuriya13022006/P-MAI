from typing import Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.ollama_provider import OllamaProvider, get_ollama_provider
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


def get_active_chat_llm(
    canonical_llm: LLMProvider = Depends(get_llm_provider),
    legacy_llm: Any = Depends(get_ollama_provider),
) -> LLMProvider:
    """Dependency resolver supporting cloud-first primary while honoring legacy mock overrides."""
    if not isinstance(legacy_llm, OllamaProvider):
        return legacy_llm
    return canonical_llm


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_active_chat_llm),
) -> ChatResponse:

    chat_service = ChatService(db, llm=llm)

    response = chat_service.chat(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        user_message=request.message,
    )

    return ChatResponse(
        conversation_id=request.conversation_id,
        response=response,
    )