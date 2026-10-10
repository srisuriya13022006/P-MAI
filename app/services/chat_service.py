from sqlalchemy.orm import Session

from app.agent.policy import (
    is_self_identity_query,
    is_pure_datetime_query,
    is_greeting_query,
    is_acknowledgment_query,
    is_farewell_query,
    is_pure_calculator_query,
)
from app.agent.orchestrator import MAIOrchestrator
from app.core.config import settings
from app.agent.task_state import TaskStateManager
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.messages import MessageRepository
from app.database.repositories.memories import MemoryRepository
from app.llm.factory import get_llm_provider
from app.llm.ollama_provider import get_ollama_provider
from app.memory.llm_extractor import LLMExtractionProvider
from app.memory.service import MemoryService
from app.memory.summarizer import ConversationSummarizer
from app.services.message_mapper import to_llm_messages

_default_task_state_manager = TaskStateManager()


class ChatService:
    """Coordinates persistent MAI conversations."""

    def __init__(
        self,
        db: Session,
        llm: LLMExtractionProvider | None = None,
        task_state_manager: TaskStateManager | None = None,
    ):
        self.db = db

        provider = llm if llm is not None else get_llm_provider()

        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        self.memory_service = MemoryService(MemoryRepository(db), llm=provider)
        self.summarizer = ConversationSummarizer()
        self.task_state_manager = task_state_manager or _default_task_state_manager

        self.orchestrator = MAIOrchestrator(
            self.memory_service,
            llm=provider,
            task_state_manager=self.task_state_manager,
        )

    def chat(
        self,
        conversation_id: str,
        user_id: str,
        user_message: str,
    ) -> str:

        if (
            self.memory_service.extractor._llm is not None
            and getattr(self.orchestrator, "llm", None) is not None
            and self.memory_service.extractor._llm.llm is not self.orchestrator.llm
        ):
            self.memory_service.extractor._llm.llm = self.orchestrator.llm

        existing_conversation = self.conversation_repo.get_by_id(
            conversation_id
        )
        is_new_conversation = (
            existing_conversation is None
            or len(self.message_repo.get_for_conversation(existing_conversation.id)) == 0
        )

        conversation = existing_conversation
        if conversation is None:
            conversation = self.conversation_repo.create(
                conversation_id=conversation_id,
                user_id=user_id,
                title=None,
            )

        # Store the new user message.
        self.message_repo.create(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )

        # Load complete conversation history.
        messages = self.message_repo.get_for_conversation(
            conversation.id
        )

        llm_messages = to_llm_messages(messages)
        if conversation.summary:
            llm_messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"Earlier conversation summary:\n{conversation.summary}",
                },
            )

        is_self_id = is_self_identity_query(user_message)
        is_dt_query = is_pure_datetime_query(user_message)
        is_calc_query = is_pure_calculator_query(user_message)
        is_farewell = is_farewell_query(user_message)
        is_casual = is_greeting_query(user_message) or is_acknowledgment_query(user_message)

        # Profile context is constructed for the initial turn of a new conversation
        # or whenever the user asks about their profile or personal memories
        personal_memory_question_starters = (
            "what do you know",
            "what do you remember",
            "tell me about myself",
            "what have you learned",
            "who am i",
            "what is my",
            "what are my",
            "what do i",
            "do i have",
            "my profile",
        )
        is_profile_or_personal_query = (
            not is_self_id
            and not is_dt_query
            and any(
                phrase in user_message.lower().strip()
                for phrase in personal_memory_question_starters
            )
        )

        user_profile_context = None
        user_profile_memories = []

        if not is_self_id and not is_dt_query and (is_new_conversation or is_profile_or_personal_query):
            user_profile_memories = self.memory_service.get_user_profile_memories(user_id)
            user_profile_context = self.memory_service.build_user_profile(user_id)

        # Per-message relevant memory retrieval (only active, non-superseded memories)
        memory_context = []
        skip_relevant_memory = is_self_id or is_dt_query or is_calc_query or is_farewell or is_casual
        if not skip_relevant_memory:
            memory_context = self.memory_service.relevant_context(
                user_id=user_id,
                query=user_message,
            )

        # Context deduplication: do not repeat profile memories in relevant memory context
        if user_profile_memories:
            profile_contents = {m.content.strip().lower() for m in user_profile_memories}
            memory_context = [
                mem for mem in memory_context
                if mem.strip().lower() not in profile_contents
            ]

        # Reset turn accounting on resilient LLM provider if supported
        if hasattr(self.orchestrator.llm, "reset_turn_accounting"):
            self.orchestrator.llm.reset_turn_accounting()

        # Let MAI decide how to handle the request.
        response = self.orchestrator.handle(
            user_message=user_message,
            conversation_messages=llm_messages,
            memory_context=memory_context,
            user_id=user_id,
            user_profile=user_profile_context,
            conversation_id=conversation.id,
        )

        # Store the assistant response.
        self.message_repo.create(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
        )

        if (
            len(messages) >= 20
            and len(messages) % settings.conversation_summary_interval == 0
        ):
            summary = self.summarizer.summarize(llm_messages[1:-1])
            self.conversation_repo.update_summary(conversation, summary)

        self.db.commit()

        return response