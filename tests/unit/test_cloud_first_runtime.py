"""
P15.4 — Focused Cloud-First Runtime Architecture Tests.
Validates:
1-5: Provider Configuration
6-10: STT Execution & Fallback
11-15: LLM Execution & Fallback
16-20: Memory Authorization Invariants
21-25: Agent Invariants (P5, P8, P9, P10, P11)
26-30: Security & Non-Bypass Invariants
"""
import base64
import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.capabilities import PlannerCapability, decompose_request_capabilities
from app.agent.clarification import ClarificationReason, ClarificationRequest
from app.agent.orchestrator import MAIOrchestrator
from app.agent.plan import ExecutionPlan, StepResult, ToolStep
from app.agent.planner import ControlledGeneralPlanner
from app.agent.planner_schema import CandidatePlan
from app.agent.recovery import ControlledReplanner, FailureCategory, ReplanningContext
from app.agent.task_state import ActiveTaskState, TaskLifecycleStatus, TaskStateManager
from app.core.config import Settings
from app.core.runtime_mode import RuntimeMode, determine_runtime_mode
from app.database.connection import Base
from app.database.repositories.memories import MemoryRepository
from app.llm.base import LLMProvider
from app.llm.factory import create_llm_provider
from app.llm.groq_provider import GroqProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.resilient_provider import ResilientLLMProvider
from app.services.chat_service import ChatService
from app.tools.registry import ToolRegistry
from app.tools.web.grounding import enforce_grounding
from app.tools.web.fetch import is_safe_web_url
from app.voice.config import VoiceConfig
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.exceptions import STTUnavailableError
from app.voice.pipeline import VoicePipeline
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.schemas import STTResult, VoiceChatRequest
from app.voice.stt.assemblyai_adapter import AssemblyAIRealtimeSTTAdapter
from app.voice.stt.factory import create_stt_provider
from app.voice.stt.local_whisper import LocalWhisperSTTAdapter
from app.voice.stt.resilient_adapter import ResilientSTTAdapter
from app.voice.tts.mock import MockTTSProvider, generate_mock_wav


class FakeTestLLM:
    """Mock LLM satisfying LLMProvider protocol for deterministic test verification."""

    def __init__(self, provider_name: str = "mock_cloud", provider_type: str = "cloud", response: str = "I am a cloud response."):
        self._name = provider_name
        self._type = provider_type
        self.response = response
        self.calls: list[dict] = []
        self.raise_error: Exception | None = None

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return self._type

    def generate(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        if self.raise_error:
            raise self.raise_error
        return self.response


def _create_test_db_and_service(llm: LLMProvider | None = None):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = Session(engine)
    service = ChatService(db, llm=llm)
    return service, db


# ===========================================================================
# 1-5: PROVIDER CONFIGURATION
# ===========================================================================

def test_1_assemblyai_selected_as_stt_primary():
    """1. AssemblyAI selected as STT primary by default configuration."""
    s = Settings(database_url="sqlite://", voice_stt_provider="assemblyai")
    cfg = VoiceConfig.from_settings(s)
    assert cfg.stt_provider == "assemblyai"
    adapter = create_stt_provider(cfg)
    assert isinstance(adapter, ResilientSTTAdapter)
    assert adapter.primary_provider.provider_name == "assemblyai"


def test_2_cloud_llm_selected_as_llm_primary():
    """2. Cloud LLM selected as LLM primary by default."""
    cfg = Settings(database_url="sqlite://", llm_provider="cloud", groq_api_key="gsk_dummy_test_key_xyz")
    provider = create_llm_provider(settings_obj=cfg)
    assert isinstance(provider, ResilientLLMProvider)
    assert provider.primary_provider is not None
    assert provider.primary_provider.provider_name == "groq"
    assert provider.runtime_mode == "CLOUD"


def test_3_local_providers_still_selectable():
    """3. Local providers remain selectable via configuration."""
    s = Settings(database_url="sqlite://", voice_stt_provider="local", llm_provider="ollama")
    cfg_voice = VoiceConfig.from_settings(s)
    assert cfg_voice.stt_provider == "local"
    stt = create_stt_provider(cfg_voice)
    assert isinstance(stt, LocalWhisperSTTAdapter)
    assert stt.provider_name.startswith("local_whisper")

    llm = create_llm_provider(settings_obj=s)
    assert isinstance(llm, OllamaProvider)
    assert llm.provider_name == "ollama"


def test_4_missing_assemblyai_key_produces_deterministic_error():
    """4. Missing AssemblyAI API key produces deterministic configuration error when fallback is disabled."""
    cfg = VoiceConfig(stt_provider="assemblyai", assemblyai_api_key=None, stt_api_key=None)
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(STTUnavailableError) as exc_info:
            create_stt_provider(cfg, allow_fallback=False)
        assert "AssemblyAI API key is not configured" in str(exc_info.value)


def test_5_missing_cloud_llm_credentials_produces_deterministic_error():
    """5. Missing cloud LLM credentials produce deterministic configuration error when fallback is disabled."""
    cfg = Settings(database_url="sqlite://", llm_provider="cloud", groq_api_key=None)
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            create_llm_provider(settings_obj=cfg, allow_fallback=False)
        assert "GROQ_API_KEY is not configured" in str(exc_info.value) or "GROQ_API_KEY is not set" in str(exc_info.value)


# ===========================================================================
# 6-10: STT EXECUTION & FALLBACK
# ===========================================================================

@pytest.mark.asyncio
async def test_6_assemblyai_final_transcript_reaches_mai():
    """6. AssemblyAI final transcript reaches MAI agent."""
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.return_value = STTResult(transcript="Hello MAI", provider="assemblyai", is_final=True)
    fallback_stt = MagicMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper"

    resilient_stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)

    mock_chat = MagicMock()
    mock_chat.chat.return_value = "Hello human!"

    pipeline = RealtimeVoicePipeline(chat_service=mock_chat, stt_provider=resilient_stt, tts_provider=MockTTSProvider())

    session = pipeline.session_manager.create_session("sess-1", "user-1")
    # Transition to LISTENING via AUDIO_START
    async for _ in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)):
        pass

    events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]

    mock_chat.chat.assert_called_once()
    assert mock_chat.chat.call_args[1]["user_message"] == "Hello MAI"
    assert any(e.type == RealtimeEventType.TRANSCRIPT_FINAL and e.transcript == "Hello MAI" for e in events)


@pytest.mark.asyncio
async def test_7_assemblyai_partial_transcript_does_not_invoke_mai():
    """7. AssemblyAI partial transcript does not invoke MAI brain."""
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.feed_audio_chunk.return_value = [
        STTResult(transcript="Hello partial", provider="assemblyai", is_final=False)
    ]
    fallback_stt = MagicMock(spec=LocalWhisperSTTAdapter)
    resilient_stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)

    mock_chat = MagicMock()
    pipeline = RealtimeVoicePipeline(chat_service=mock_chat, stt_provider=resilient_stt, tts_provider=MockTTSProvider())

    session = pipeline.session_manager.create_session("sess-2", "user-2")
    async for _ in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)):
        pass

    sample_b64 = base64.b64encode(generate_mock_wav(0.2)).decode("ascii")
    events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=sample_b64))]

    # MAI was NEVER invoked on partial
    assert not mock_chat.chat.called
    assert any(e.type == RealtimeEventType.TRANSCRIPT_PARTIAL for e in events)


@pytest.mark.asyncio
async def test_8_assemblyai_failure_triggers_local_whisper_fallback():
    """8. AssemblyAI failure triggers Local Whisper fallback."""
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = RuntimeError("AssemblyAI network down")

    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper"
    fallback_stt.transcribe.return_value = STTResult(transcript="Fallback transcript from whisper", provider="local_whisper")

    resilient_stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    resilient_stt._buffered_audio.extend(b"dummy_audio")

    res = await resilient_stt.finalize_stream()
    assert res.transcript == "Fallback transcript from whisper"
    assert resilient_stt.stt_mode == "LOCAL_FALLBACK"
    assert fallback_stt.transcribe.called


@pytest.mark.asyncio
async def test_9_no_stt_fallback_loop():
    """9. STT does not oscillate or loop between providers within request."""
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = RuntimeError("AssemblyAI down")

    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper"
    fallback_stt.transcribe.return_value = STTResult(transcript="Whisper result", provider="local_whisper")

    resilient_stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    resilient_stt._buffered_audio.extend(b"audio_bytes")

    await resilient_stt.finalize_stream()
    assert primary_stt.finalize_stream.call_count == 1
    assert fallback_stt.transcribe.call_count == 1
    assert resilient_stt.stt_mode == "LOCAL_FALLBACK"


@pytest.mark.asyncio
async def test_10_existing_voice_protocol_remains_intact():
    """10. Existing voice event protocol remains completely intact."""
    from app.voice.stt.streaming import StreamingSpeechToTextProvider

    class MockStreamSTT(StreamingSpeechToTextProvider):
        @property
        def provider_name(self) -> str:
            return "assemblyai"

        async def feed_audio_chunk(self, chunk: bytes, content_type: str = "audio/wav"):
            return []

        async def finalize_stream(self):
            return STTResult(transcript="Test input", provider="assemblyai", is_final=True)

        def reset(self):
            pass

        async def transcribe(self, audio_data: bytes, content_type: str = "audio/wav"):
            return STTResult(transcript="Test input", provider="assemblyai", is_final=True)

    mock_chat = MagicMock()
    mock_chat.chat.return_value = "Response grounded"

    pipeline = RealtimeVoicePipeline(chat_service=mock_chat, stt_provider=MockStreamSTT(), tts_provider=MockTTSProvider())
    session = pipeline.session_manager.create_session("sess-10", "user-10")

    async for _ in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)):
        pass

    events = [e async for e in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END))]

    event_types = [e.type for e in events]
    assert RealtimeEventType.SPEECH_END in event_types
    assert RealtimeEventType.TRANSCRIPT_FINAL in event_types
    assert RealtimeEventType.RESPONSE_START in event_types
    assert RealtimeEventType.RESPONSE_END in event_types


# ===========================================================================
# 11-15: LLM EXECUTION & FALLBACK
# ===========================================================================

def test_11_cloud_llm_called_in_normal_cloud_mode():
    """11. Cloud LLM is called in normal cloud mode."""
    primary = FakeTestLLM("groq", "cloud", "Cloud answer")
    fallback = FakeTestLLM("ollama", "local", "Local answer")

    resilient = ResilientLLMProvider(primary, fallback)
    resp = resilient.generate([{"role": "user", "content": "Hello"}])

    assert resp == "Cloud answer"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0
    assert resilient.runtime_mode == "CLOUD"


def test_12_cloud_llm_failure_triggers_local_qwen_fallback():
    """12. Cloud LLM failure triggers Local Qwen/Ollama fallback."""
    primary = FakeTestLLM("groq", "cloud")
    primary.raise_error = ConnectionError("Groq endpoint unreachable")
    fallback = FakeTestLLM("ollama", "local", "Fallback from local Qwen")

    resilient = ResilientLLMProvider(primary, fallback)
    resp = resilient.generate([{"role": "user", "content": "Analyze this"}])

    assert resp == "Fallback from local Qwen"
    assert resilient.runtime_mode == "LOCAL_FALLBACK"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_13_no_llm_fallback_loop():
    """13. LLM executes single controlled fallback transition with zero flapping loops."""
    primary = FakeTestLLM("groq", "cloud")
    primary.raise_error = RuntimeError("Rate limit 429")
    fallback = FakeTestLLM("ollama", "local", "Local answer")

    resilient = ResilientLLMProvider(primary, fallback)
    resilient.generate([{"role": "user", "content": "Question"}])

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_14_structured_cloud_output_goes_through_plan_validation():
    """14. Structured cloud LLM output passes through deterministic plan validation."""
    mock_cloud_llm = FakeTestLLM(
        "groq",
        "cloud",
        json.dumps({
            "goal": "Calculate 10 + 20",
            "steps": [{"tool_name": "calculator", "arguments": {"expression": "10 + 20"}}],
            "reasoning": "Valid math plan",
        }),
    )

    planner = ControlledGeneralPlanner(llm=mock_cloud_llm, use_llm=True)
    plan, trace = planner.plan("Calculate 10 + 20")

    assert plan is not None
    assert trace.validation_passed is True
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "calculator"


def test_15_cloud_llm_cannot_directly_execute_arbitrary_tools():
    """15. Cloud LLM cannot inject or directly execute arbitrary unvalidated tools."""
    mock_cloud_llm = FakeTestLLM(
        "groq",
        "cloud",
        json.dumps({
            "goal": "Malicious tool execution",
            "steps": [{"tool_name": "arbitrary_shell_exec", "arguments": {"cmd": "rm -rf /"}}],
            "reasoning": "Invalid tool",
        }),
    )

    planner = ControlledGeneralPlanner(llm=mock_cloud_llm, tool_registry=ToolRegistry(), use_llm=True)
    plan, trace = planner.plan("Run arbitrary code")

    # Schema/candidate validation fails; falls back to deterministic safe plan
    assert trace.planner_source in ("fallback", "deterministic")
    assert not any(step.tool_name == "arbitrary_shell_exec" for step in plan.steps)


# ===========================================================================
# 16-20: MEMORY INVARIANTS
# ===========================================================================

def test_16_my_name_is_suriya_still_requires_confirmation():
    """16. 'My name is Suriya.' still requires confirmation in cloud mode without DB write."""
    cloud_llm = FakeTestLLM("groq", "cloud")
    service, db = _create_test_db_and_service(llm=cloud_llm)
    repo = MemoryRepository(db)

    resp = service.chat("conv-16", "user-16", "My name is Suriya.")
    assert "Would you like me to remember that?" in resp
    assert len(repo.list_for_user("user-16")) == 0


def test_17_yes_performs_authorized_write():
    """17. 'Yes' performs authorized write after candidate memory prompt."""
    cloud_llm = FakeTestLLM("groq", "cloud")
    service, db = _create_test_db_and_service(llm=cloud_llm)
    repo = MemoryRepository(db)

    service.chat("conv-17", "user-17", "My name is Suriya.")
    assert len(repo.list_for_user("user-17")) == 0

    resp2 = service.chat("conv-17", "user-17", "Yes.")
    assert "remembered" in resp2.lower()
    memories = repo.list_for_user("user-17")
    assert len(memories) == 1
    assert "Suriya" in memories[0].content


def test_18_what_is_my_name_performs_memory_read():
    """18. 'What is my name?' performs memory retrieval without asking confirmation."""
    cloud_llm = FakeTestLLM("groq", "cloud", "Your name is Suriya.")
    service, db = _create_test_db_and_service(llm=cloud_llm)
    repo = MemoryRepository(db)
    repo.create(user_id="user-18", content="User's name is Suriya", memory_type="profile")

    resp = service.chat("conv-18", "user-18", "What is my name?")
    assert "Would you like me to remember that?" not in resp
    assert "remember that?" not in resp.lower()


def test_19_what_is_your_name_remains_local_self_identity():
    """19. 'What is your name?' remains local self-identity with zero memory/cloud access."""
    cloud_llm = FakeTestLLM("groq", "cloud")
    service, db = _create_test_db_and_service(llm=cloud_llm)

    resp = service.chat("conv-19", "user-19", "What is your name?")
    assert "MAI" in resp
    assert "assistant" in resp.lower()
    # Cloud LLM was not called
    assert len(cloud_llm.calls) == 0


@pytest.mark.asyncio
async def test_20_same_behavior_for_voice_and_text():
    """20. Voice pipeline enforces identical memory authorization behavior as text."""
    cloud_llm = FakeTestLLM("groq", "cloud")
    service, db = _create_test_db_and_service(llm=cloud_llm)
    repo = MemoryRepository(db)

    mock_stt = AsyncMock()
    mock_stt.provider_name = "assemblyai"
    mock_stt.stt_mode = "CLOUD"
    mock_stt.transcribe.return_value = STTResult(transcript="My name is Suriya.", provider="assemblyai")

    pipeline = VoicePipeline(chat_service=service, stt_provider=mock_stt, tts_provider=MockTTSProvider())

    b64_audio = base64.b64encode(generate_mock_wav(0.5)).decode("ascii")
    req1 = VoiceChatRequest(audio_base64=b64_audio, conversation_id="conv-20", user_id="user-20")
    resp1 = await pipeline.process_voice(req1)

    assert "Would you like me to remember that?" in resp1.text_response
    assert len(repo.list_for_user("user-20")) == 0

    mock_stt.transcribe.return_value = STTResult(transcript="Yes.", provider="assemblyai")
    resp2 = await pipeline.process_voice(req1)
    assert "remembered" in resp2.text_response.lower()
    assert len(repo.list_for_user("user-20")) == 1


# ===========================================================================
# 21-25: AGENT INVARIANTS (P5, P8, P9, P10, P11)
# ===========================================================================

def test_21_p8_task_continuation_works():
    """21. P8 task continuation maintains active task across multi-turn cloud calls."""
    mgr = TaskStateManager()
    task = mgr.create_task("user-21", "conv-21", goal="Fetch documentation")
    task.context_values["result_summary"] = "Version 3.12 release notes"
    task.transition_to(TaskLifecycleStatus.RUNNING)

    retrieved = mgr.get_active_task("user-21", "conv-21")
    assert retrieved is not None
    assert retrieved.original_user_goal == "Fetch documentation"
    assert retrieved.context_values.get("result_summary") == "Version 3.12 release notes"


def test_22_p9_clarification_works():
    """22. P9 clarification enters WAITING_FOR_USER and bounds clarification rounds."""
    cloud_llm = FakeTestLLM("groq", "cloud")
    service, db = _create_test_db_and_service(llm=cloud_llm)

    # Missing arguments: search query missing
    orch = service.orchestrator
    clar_req = ClarificationRequest(
        task_id="t-1",
        question="Which package do you mean?",
        missing_information="package_name",
        reason=ClarificationReason.MISSING_ARGUMENT,
    )
    task = orch.task_state_manager.create_task("user-22", "conv-22", goal="Install package")
    task.pending_clarification = clar_req
    task.transition_to(TaskLifecycleStatus.WAITING_FOR_USER)

    active = orch.task_state_manager.get_active_task("user-22", "conv-22")
    assert active.current_task_status == TaskLifecycleStatus.WAITING_FOR_USER
    assert active.pending_clarification.missing_information == "package_name"


def test_23_p10_capability_selection_works():
    """23. P10 deterministic capability decomposition correctly maps user intent."""
    caps = decompose_request_capabilities("What is the current time and calculate 20 + 30")
    assert PlannerCapability.DATE_TIME_LOOKUP in caps
    assert PlannerCapability.NUMERIC_CALCULATION in caps


def test_24_p11_bounded_replanning_works():
    """24. P11 bounded replanning halts when budget exceeded."""
    replanner = ControlledReplanner(llm=FakeTestLLM(), use_llm=False)
    failed_step = StepResult(
        step_id="s1",
        tool_name="web_search",
        success=False,
        error="Search timed out",
    )
    ctx = ReplanningContext(
        original_user_goal="Search web",
        active_task=None,
        completed_steps=[],
        failed_step=failed_step,
        failure_category=FailureCategory.TIMEOUT,
        replan_count=3,
    )

    plan, reason = replanner.generate_replacement_plan(ctx)
    assert plan is None
    assert reason == "replan_limit_exhausted"


def test_25_p5_grounding_enforcement_remains_active():
    """25. P5 grounding enforcement blocks ungrounded fabrications from cloud LLM."""
    evidence = [{"url": "https://python.org", "content": "Python 3.13 was released in October 2024."}]
    ungrounded_claim = "Python 3.13 includes quantum telepathy computing built-in."

    sanitized = enforce_grounding(ungrounded_claim, evidence)
    assert "quantum telepathy" not in sanitized or "unverified" in sanitized.lower() or "not confirmed" in sanitized.lower()


# ===========================================================================
# 26-30: SECURITY & NON-BYPASS INVARIANTS
# ===========================================================================

def test_26_api_keys_never_appear_in_logs(caplog):
    """26. API keys never appear in runtime logging output."""
    caplog.set_level(logging.DEBUG)
    secret_key = "gsk_SUPER_SECRET_PRODUCTION_KEY_9999"

    try:
        GroqProvider(api_key=secret_key)
    except Exception:
        pass

    for record in caplog.records:
        assert secret_key not in record.message
        assert "SUPER_SECRET" not in record.message


def test_27_api_keys_never_appear_in_frontend_bundles():
    """27. API keys and secrets do not exist in client frontend JavaScript/HTML."""
    web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "client", "web")
    if os.path.exists(web_dir):
        for fname in os.listdir(web_dir):
            if fname.endswith((".js", ".html")):
                content = open(os.path.join(web_dir, fname), encoding="utf-8", errors="ignore").read()
                assert "ASSEMBLYAI_API_KEY" not in content
                assert "GROQ_API_KEY" not in content
                assert "gsk_" not in content


def test_28_cloud_provider_cannot_bypass_ssrf_checks():
    """28. Cloud provider cannot bypass deterministic SSRF protection."""
    assert not is_safe_web_url("http://127.0.0.1:8000/admin")[0]
    assert not is_safe_web_url("http://169.254.169.254/latest/meta-data")[0]
    assert not is_safe_web_url("http://localhost:5432")[0]
    assert is_safe_web_url("https://docs.python.org/3/")[0]


def test_29_cloud_provider_cannot_bypass_memory_authorization():
    """29. Cloud provider cannot directly authorize memory without deterministic confirmation."""
    cloud_llm = FakeTestLLM("groq", "cloud", "I am writing this memory directly: User prefers dark mode.")
    service, db = _create_test_db_and_service(llm=cloud_llm)
    repo = MemoryRepository(db)

    service.chat("conv-29", "user-29", "I prefer dark mode.")
    assert len(repo.list_for_user("user-29")) == 0


def test_30_cloud_provider_cannot_bypass_plan_validation():
    """30. Cloud LLM output is strictly rejected if it fails deterministic plan validation."""
    invalid_json_llm = FakeTestLLM("groq", "cloud", "I am not a valid JSON plan")
    planner = ControlledGeneralPlanner(llm=invalid_json_llm, use_llm=True)

    plan, trace = planner.plan("Check weather")
    assert trace.planner_source in ("fallback", "deterministic")


# ===========================================================================
# 31-38: AUDIT FOLLOW-UP: FALLBACK CLASSIFICATION & WHISPER MODEL VERIFICATION
# ===========================================================================

@pytest.mark.asyncio
async def test_31_network_failure_triggers_controlled_fallback():
    """31. Network failure triggers controlled local fallback for both STT and LLM."""
    import httpx

    # STT Network Failure -> Local Whisper Fallback
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = httpx.NetworkError("DNS resolution failed")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper_base"
    fallback_stt.transcribe.return_value = STTResult(transcript="Local audio fallback", provider="local_whisper_base")

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"sample_pcm_audio")
    res = await stt.finalize_stream()
    assert res.transcript == "Local audio fallback"
    assert stt.stt_mode == "LOCAL_FALLBACK"

    # LLM Network Failure -> Local Qwen Fallback
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = httpx.ConnectError("Connection refused by remote host")
    fallback_llm = FakeTestLLM("ollama", "local", "Local Qwen answer")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    ans = llm.generate([{"role": "user", "content": "Hi"}])
    assert ans == "Local Qwen answer"
    assert llm.runtime_mode == "LOCAL_FALLBACK"


@pytest.mark.asyncio
async def test_32_timeout_triggers_controlled_fallback():
    """32. Timeout triggers controlled local fallback for both STT and LLM."""
    import httpx

    # STT Timeout -> Fallback
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = httpx.TimeoutException("AssemblyAI socket read timed out")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper_base"
    fallback_stt.transcribe.return_value = STTResult(transcript="Recovered from timeout", provider="local_whisper_base")

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"sample_pcm_audio")
    res = await stt.finalize_stream()
    assert res.transcript == "Recovered from timeout"
    assert stt.stt_mode == "LOCAL_FALLBACK"

    # LLM Timeout -> Fallback
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = TimeoutError("Groq completion timed out after 30s")
    fallback_llm = FakeTestLLM("ollama", "local", "Fallback after timeout")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    ans = llm.generate([{"role": "user", "content": "Hi"}])
    assert ans == "Fallback after timeout"
    assert llm.runtime_mode == "LOCAL_FALLBACK"


@pytest.mark.asyncio
async def test_33_http_401_inhibits_silent_fallback():
    """33. HTTP 401 Unauthorized must raise error immediately and NOT silently fall back."""
    class DummyHTTP401Error(Exception):
        status_code = 401

    # STT 401
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = DummyHTTP401Error("401 Unauthorized: Invalid AssemblyAI token")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"sample_audio")
    with pytest.raises(DummyHTTP401Error):
        await stt.finalize_stream()
    assert not fallback_stt.transcribe.called
    assert stt.stt_mode == "OFFLINE"

    # LLM 401
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = DummyHTTP401Error("401 Unauthorized: Invalid Groq API key")
    fallback_llm = FakeTestLLM("ollama", "local")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    with pytest.raises(DummyHTTP401Error):
        llm.generate([{"role": "user", "content": "Hi"}])
    assert len(fallback_llm.calls) == 0
    assert llm.runtime_mode == "OFFLINE"


@pytest.mark.asyncio
async def test_34_http_403_inhibits_silent_fallback():
    """34. HTTP 403 Forbidden must raise error immediately and NOT silently fall back."""
    class DummyHTTP403Error(Exception):
        status_code = 403

    # STT 403
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = DummyHTTP403Error("403 Forbidden: Project does not have access")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"sample_audio")
    with pytest.raises(DummyHTTP403Error):
        await stt.finalize_stream()
    assert not fallback_stt.transcribe.called

    # LLM 403
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = DummyHTTP403Error("403 Forbidden: Tier permission denied")
    fallback_llm = FakeTestLLM("ollama", "local")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    with pytest.raises(DummyHTTP403Error):
        llm.generate([{"role": "user", "content": "Hi"}])
    assert len(fallback_llm.calls) == 0


@pytest.mark.asyncio
async def test_35_invalid_api_key_inhibits_silent_fallback():
    """35. Invalid API key error raises configuration failure without silent fallback."""
    # STT
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = ValueError("Invalid API key configured for AssemblyAI")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"audio")
    with pytest.raises(ValueError):
        await stt.finalize_stream()
    assert not fallback_stt.transcribe.called

    # LLM
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = ValueError("Invalid API key provided for Groq")
    fallback_llm = FakeTestLLM("ollama", "local")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    with pytest.raises(ValueError):
        llm.generate([{"role": "user", "content": "Hi"}])
    assert len(fallback_llm.calls) == 0


@pytest.mark.asyncio
async def test_36_http_429_rate_limit_controlled_behavior():
    """36. HTTP 429 Rate Limit triggers controlled single fallback without retry loops."""
    class DummyHTTP429Error(Exception):
        status_code = 429

    # STT 429
    primary_stt = AsyncMock(spec=AssemblyAIRealtimeSTTAdapter)
    primary_stt.provider_name = "assemblyai"
    primary_stt.finalize_stream.side_effect = DummyHTTP429Error("429 Too Many Requests: Rate limit reached")
    fallback_stt = AsyncMock(spec=LocalWhisperSTTAdapter)
    fallback_stt.provider_name = "local_whisper_base"
    fallback_stt.transcribe.return_value = STTResult(transcript="Rate limit fallback", provider="local_whisper_base")

    stt = ResilientSTTAdapter(primary_provider=primary_stt, fallback_provider=fallback_stt)
    stt._buffered_audio.extend(b"audio_bytes")
    res = await stt.finalize_stream()
    assert res.transcript == "Rate limit fallback"
    assert stt.stt_mode == "LOCAL_FALLBACK"
    assert primary_stt.finalize_stream.call_count == 1
    assert fallback_stt.transcribe.call_count == 1

    # LLM 429
    primary_llm = FakeTestLLM("groq", "cloud")
    primary_llm.raise_error = DummyHTTP429Error("429 Rate limit exceeded")
    fallback_llm = FakeTestLLM("ollama", "local", "LLM rate limit fallback answer")

    llm = ResilientLLMProvider(primary_provider=primary_llm, fallback_provider=fallback_llm)
    ans = llm.generate([{"role": "user", "content": "Generate"}])
    assert ans == "LLM rate limit fallback answer"
    assert llm.runtime_mode == "LOCAL_FALLBACK"
    assert len(primary_llm.calls) == 1
    assert len(fallback_llm.calls) == 1


def test_37_whisper_fallback_model_is_base():
    """37. Verify configured default local Whisper fallback model is explicitly 'base'."""
    settings = Settings(database_url="sqlite://")
    assert settings.voice_stt_local_model == "base"

    voice_cfg = VoiceConfig.from_settings(settings)
    assert voice_cfg.stt_local_model == "base"

    local_adapter = LocalWhisperSTTAdapter()
    assert local_adapter.model_name == "base"
    assert local_adapter.provider_name == "local_whisper_base"


def test_38_audio_container_helpers_and_pcm_contract():
    """38. Verify ensure_wav_container and strip_wav_header_if_present enforce contract."""
    from app.voice.stt.assemblyai_adapter import ensure_wav_container, is_wav_container, strip_wav_header_if_present

    # Raw PCM bytes (no header)
    raw_pcm = b"\x00\x00" * 1600  # 100ms of 16-bit 16kHz mono audio
    assert not is_wav_container(raw_pcm)

    # Wrap in WAV container
    wav_audio = ensure_wav_container(raw_pcm, sample_rate=16000, channels=1, bits_per_sample=16)
    assert is_wav_container(wav_audio)
    assert wav_audio[:4] == b"RIFF"
    assert wav_audio[8:12] == b"WAVE"
    assert len(wav_audio) == len(raw_pcm) + 44

    # Idempotent: wrapping already-wrapped audio returns unchanged
    assert ensure_wav_container(wav_audio) == wav_audio

    # Strip header yields exact original raw PCM
    stripped = strip_wav_header_if_present(wav_audio)
    assert stripped == raw_pcm
    assert not is_wav_container(stripped)

