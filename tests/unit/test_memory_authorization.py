"""
P15.2 — Memory Authorization Tests.
Validates that memory writes (both explicit commands and user personal facts)
require explicit confirmation before persisting to the database.
"""
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import pytest

from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.database.repositories.conversations import ConversationRepository
from app.database.repositories.messages import MessageRepository
from app.services.chat_service import ChatService
from app.voice.pipeline import VoicePipeline
from app.voice.schemas import VoiceChatRequest
from app.voice.config import VoiceConfig
from app.voice.stt.mock import MockSTTProvider
from app.voice.tts.mock import MockTTSProvider
from app.voice.vad.detector import EnergyVADDetector
from app.agent.task_state import TaskLifecycleStatus


def _create_chat_service():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = Session(engine)
    service = ChatService(db)
    return service, db


def test_scenario_a_personal_fact_asks_confirmation_no_db_write():
    """A. 'My name is Suriya.' -> asks confirmation -> no DB write occurs."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp = service.chat(
        conversation_id="conv-a",
        user_id="user-a",
        user_message="My name is Suriya.",
    )

    assert "remember" in resp.lower()
    assert "?" in resp
    # No memories written to database
    memories = repo.list_for_user("user-a")
    assert len(memories) == 0


def test_scenario_b_positive_confirmation_persists_memory():
    """B. 'Yes.' after A -> DB write occurs."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp1 = service.chat(
        conversation_id="conv-b",
        user_id="user-b",
        user_message="My name is Suriya.",
    )
    assert "?" in resp1
    assert len(repo.list_for_user("user-b")) == 0

    resp2 = service.chat(
        conversation_id="conv-b",
        user_id="user-b",
        user_message="Yes.",
    )
    assert "remembered" in resp2.lower()

    # DB write has occurred
    memories = repo.list_for_user("user-b")
    assert len(memories) == 1
    assert memories[0].content == "User's name is Suriya"
    assert memories[0].is_active is True


def test_scenario_c_negative_confirmation_discards_memory():
    """C. 'No.' after A -> no DB write occurs."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp1 = service.chat(
        conversation_id="conv-c",
        user_id="user-c",
        user_message="My name is Suriya.",
    )
    assert "?" in resp1
    assert len(repo.list_for_user("user-c")) == 0

    resp2 = service.chat(
        conversation_id="conv-c",
        user_id="user-c",
        user_message="No.",
    )
    assert "won't remember" in resp2.lower() or "understood" in resp2.lower()

    # DB write did NOT occur
    memories = repo.list_for_user("user-c")
    assert len(memories) == 0


def test_scenario_d_explicit_write_request_requires_confirmation():
    """D. 'Remember that my name is Suriya.' -> asks confirmation -> no DB write until confirmed."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp1 = service.chat(
        conversation_id="conv-d",
        user_id="user-d",
        user_message="Remember that my name is Suriya.",
    )
    assert "confirmation" in resp1.lower() or "remember" in resp1.lower()
    assert len(repo.list_for_user("user-d")) == 0

    resp2 = service.chat(
        conversation_id="conv-d",
        user_id="user-d",
        user_message="Yes, please.",
    )
    assert "remembered" in resp2.lower()

    memories = repo.list_for_user("user-d")
    assert len(memories) == 1
    assert "Suriya" in memories[0].content


def test_scenario_e_memory_read_does_not_ask_confirmation():
    """E. 'What is my name?' -> memory read, no confirmation prompt."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)
    # Pre-populate confirmed memory
    service.memory_service.remember("user-e", "User's name is Suriya")
    db.commit()

    resp = service.chat(
        conversation_id="conv-e",
        user_id="user-e",
        user_message="What is my name?",
    )

    # Must NOT ask confirmation
    assert "would you like me to remember" not in resp.lower()
    assert "need your confirmation" not in resp.lower()


def test_scenario_f_assistant_identity_remains_local_no_memory():
    """F. 'What is your name?' -> local self-identity response, no memory access."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp = service.chat(
        conversation_id="conv-f",
        user_id="user-f",
        user_message="What is your name?",
    )

    assert "I'm MAI" in resp or "MAI" in resp
    assert "would you like me to remember" not in resp.lower()
    assert len(repo.list_for_user("user-f")) == 0


@pytest.mark.asyncio
async def test_scenario_g_voice_chat_pipeline_shares_same_gate():
    """G. Same A/B/C behavior through the voice chat path."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    config = VoiceConfig(
        stt_provider="mock",
        tts_provider="mock",
        vad_silence_threshold_db=-50.0,
    )
    stt = MockSTTProvider()
    tts = MockTTSProvider()
    vad = EnergyVADDetector(silence_threshold_db=-50.0)

    voice_pipeline = VoicePipeline(
        config=config,
        chat_service=service,
        stt_provider=stt,
        tts_provider=tts,
        vad_detector=vad,
    )

    # 1. Voice turn: Personal fact -> asks confirmation
    stt.enqueue_transcript("My name is Suriya.")
    # Valid 1-second 16kHz PCM WAV audio with RIFF header
    import base64
    import io
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{16000}h", *([1000] * 16000)))
    b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")
    req1 = VoiceChatRequest(
        audio_base64=b64_audio,
        conversation_id="voice-conv-1",
        user_id="voice-user-1",
    )
    v_resp1 = await voice_pipeline.process_voice(req1)
    assert "remember" in v_resp1.text_response.lower()
    assert len(repo.list_for_user("voice-user-1")) == 0

    # 2. Voice turn: Positive confirmation -> persists memory
    stt.enqueue_transcript("Yes.")
    req2 = VoiceChatRequest(
        audio_base64=b64_audio,
        conversation_id="voice-conv-1",
        user_id="voice-user-1",
    )
    v_resp2 = await voice_pipeline.process_voice(req2)
    assert "remembered" in v_resp2.text_response.lower()
    assert len(repo.list_for_user("voice-user-1")) == 1


def test_scenario_h_repeated_yes_does_not_duplicate_memory():
    """H. Repeated 'Yes' after confirmation does not duplicate the memory."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    service.chat("conv-h", "user-h", "My name is Suriya.")
    service.chat("conv-h", "user-h", "Yes.")
    assert len(repo.list_for_user("user-h")) == 1

    # Second 'Yes' in follow-up turn
    service.chat("conv-h", "user-h", "Yes.")
    assert len(repo.list_for_user("user-h")) == 1


def test_scenario_i_unrelated_request_does_not_authorize_pending_memory():
    """I. Unrelated request while WAITING_FOR_USER does not accidentally authorize the pending memory."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp1 = service.chat("conv-i", "user-i", "My name is Suriya.")
    assert "?" in resp1
    assert len(repo.list_for_user("user-i")) == 0

    # Unrelated query instead of confirming
    resp2 = service.chat("conv-i", "user-i", "Calculate 20 + 30")
    assert "50" in resp2

    # Memory must NOT have been saved
    assert len(repo.list_for_user("user-i")) == 0


def test_scenario_j_memory_supersession_after_authorized_write():
    """J. Existing memory supersession/versioning still works after an authorized write."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    # 1. Authorize initial preference: Python
    service.chat("conv-j", "user-j", "I prefer Python.")
    service.chat("conv-j", "user-j", "Yes.")
    memories1 = repo.list_for_user("user-j", include_inactive=True)
    assert len(memories1) == 1
    assert memories1[0].is_active is True
    assert "Python" in memories1[0].content

    # 2. Authorize conflicting preference: C++
    service.chat("conv-j", "user-j", "I prefer C++.")
    service.chat("conv-j", "user-j", "Yes.")
    all_memories = repo.list_for_user("user-j", include_inactive=True)
    active_memories = repo.list_for_user("user-j", include_inactive=False)

    # Exactly one active memory
    assert len(active_memories) == 1
    assert "C++" in active_memories[0].content

    # Prior memory is superseded
    assert len(all_memories) == 2
    superseded = [m for m in all_memories if not m.is_active]
    assert len(superseded) == 1
    assert "Python" in superseded[0].content


def test_personal_facts_live_in_chennai_and_use_python():
    """Verify 'I live in Chennai.' and 'I use Python now.' trigger confirmation without DB write."""
    service, db = _create_chat_service()
    repo = MemoryRepository(db)

    resp1 = service.chat("conv-k", "user-k", "I live in Chennai.")
    assert "Would you like me to remember that?" in resp1
    assert len(repo.list_for_user("user-k")) == 0

    service.chat("conv-k", "user-k", "Sure, save that.")
    assert len(repo.list_for_user("user-k")) == 1
    assert "Chennai" in repo.list_for_user("user-k")[0].content
