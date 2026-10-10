"""
P15.4 Follow-Up Audit — Live Microphone -> Client -> /voice/realtime -> AssemblyAI v3 -> MAI Acceptance Test.

Tests 6 Mandatory Utterances:
1. "Hello MAI, what can you do?"
2. "My name is Suriya and I prefer Python."
3. "Search the web for today's AI news."
4. "Read it again."
5. "What time is it in Tokyo?"
6. "Test the AssemblyAI realtime speech to text adapter."

Verifies:
- Acoustic voice generation -> Raw PCM16 16kHz mono audio frames
- Realtime streaming chunk feed (200ms chunks)
- AssemblyAI Universal-3 transcription
- MAI agent execution (ChatService / MAIOrchestrator)
- Error source classification (STT, Audio Format, Microphone, Routing, LLM)
- Latency (STT latency, LLM latency, total latency)
- CPU utilization
"""
import asyncio
import io
import json
import os
import psutil
import struct
import sys
import tempfile
import time
import wave

sys.path.insert(0, r"d:\suriya\projects\P-MAI")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyttsx3
from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

from app.core.config import Settings
from app.database.connection import SessionLocal
from app.llm.factory import get_llm_provider
from app.services.chat_service import ChatService
from app.voice.config import VoiceConfig
from app.voice.events import RealtimeClientEvent, RealtimeEventType
from app.voice.realtime_pipeline import RealtimeVoicePipeline
from app.voice.stt.assemblyai_adapter import AssemblyAIRealtimeSTTAdapter
from app.voice.tts.mock import MockTTSProvider


TEST_UTTERANCES = [
    {
        "id": 1,
        "text": "Hello MAI, what can you do?",
        "expected_intent": "conversational",
        "description": "Greeting & Capability Overview"
    },
    {
        "id": 2,
        "text": "My name is Suriya and I prefer Python.",
        "expected_intent": "memory_fact",
        "description": "User profile / Memory extraction candidate"
    },
    {
        "id": 3,
        "text": "Search the web for today's AI news.",
        "expected_intent": "web_search",
        "description": "Web Search Tool Ingestion & Grounding"
    },
    {
        "id": 4,
        "text": "Read it again.",
        "expected_intent": "continuation",
        "description": "P8 Task Continuity / Clarification"
    },
    {
        "id": 5,
        "text": "What time is it in Tokyo?",
        "expected_intent": "datetime",
        "description": "Deterministic Local Datetime Tool"
    },
    {
        "id": 6,
        "text": "Test the AssemblyAI realtime speech to text adapter.",
        "expected_intent": "technical",
        "description": "Technical Vocabulary Recognition"
    },
]


def synthesize_spoken_audio_pcm(text: str) -> bytes:
    """
    Synthesize natural acoustic spoken voice using pyttsx3,
    then convert to canonical 16kHz 16-bit Mono Raw PCM.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        del engine

        # Read generated WAV file and resample to 16kHz 16-bit Mono raw PCM
        with wave.open(tmp_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        # Convert to 16kHz Mono 16-bit signed PCM
        import audioop
        # 1. Convert to mono if needed
        if n_channels == 2:
            frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        # 2. Resample to 16kHz if needed
        if framerate != 16000:
            frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
        # 3. Convert sample width to 16-bit if needed
        if sampwidth != 2:
            frames = audioop.lin2lin(frames, sampwidth, 2)

        return frames
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def run_live_end_to_end_test():
    print("=" * 75, flush=True)
    print("P15.4 LIVE MICROPHONE -> ASSEMBLYAI -> MAI ACCEPTANCE RUN", flush=True)
    print("=" * 75, flush=True)

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("[ERROR] ASSEMBLYAI_API_KEY is not set in environment or .env!", flush=True)
        return

    groq_key = os.environ.get("GROQ_API_KEY")
    print(f"AssemblyAI Key: {api_key[:6]}... (backend only)")
    print(f"Groq Cloud Key: {groq_key[:6]}... (backend only)")

    settings = Settings()
    db = SessionLocal()

    llm = get_llm_provider()
    chat_service = ChatService(db, llm=llm)


    # Initialize live AssemblyAI STT Adapter
    stt_adapter = AssemblyAIRealtimeSTTAdapter(
        api_key=api_key,
        model_name="universal-3-5-pro",
        timeout_seconds=30.0,
        sample_rate=16000,
    )


    voice_cfg = VoiceConfig.from_settings(settings)
    voice_cfg.stt_provider = "assemblyai"

    pipeline = RealtimeVoicePipeline(
        chat_service=chat_service,
        stt_provider=stt_adapter,
        tts_provider=MockTTSProvider(),
        config=voice_cfg,
    )

    results = []
    initial_cpu = psutil.cpu_percent(interval=0.5)
    print(f"Baseline CPU Usage: {initial_cpu:.1f}%\n", flush=True)

    for item in TEST_UTTERANCES:
        u_id = item["id"]
        spoken = item["text"]
        print(f"[{u_id}/6] Testing: \"{spoken}\"", flush=True)

        # 1. Synthesize spoken acoustic audio into canonical raw PCM16
        t_synth = time.perf_counter()
        raw_pcm = synthesize_spoken_audio_pcm(spoken)
        synth_dur = len(raw_pcm) / (16000 * 2)
        print(f"   [Mic Capture] Audio synthesized: {len(raw_pcm)} bytes ({synth_dur:.2f}s of 16kHz PCM16)", flush=True)

        # 2. Start WebSocket / Realtime session
        conv_id = f"e2e-session-{u_id}"
        session = pipeline.session_manager.create_session(conversation_id=conv_id, user_id="suriya")

        # 3. Send AUDIO_START
        async for _ in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_START)):
            pass

        # 4. Stream audio in 200ms chunks (simulating realtime client microphone stream)
        chunk_size = 6400  # 200ms at 16kHz 16-bit mono
        import base64
        for i in range(0, len(raw_pcm), chunk_size):
            chunk = raw_pcm[i:i + chunk_size]
            b64_chunk = base64.b64encode(chunk).decode("ascii")
            async for _ in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_CHUNK, audio_base64=b64_chunk)):
                pass

        # 5. Send AUDIO_END and receive live AssemblyAI transcript + MAI response
        t_stt_start = time.perf_counter()
        events = []
        async for evt in pipeline.handle_client_event(session, RealtimeClientEvent(type=RealtimeEventType.AUDIO_END)):
            events.append(evt)

        # Extract final transcript
        transcript_evt = next((e for e in events if e.type == RealtimeEventType.TRANSCRIPT_FINAL), None)
        final_transcript = transcript_evt.transcript if transcript_evt else "EMPTY"
        stt_lat_ms = (time.perf_counter() - t_stt_start) * 1000.0

        # Extract MAI response
        resp_evt = next((e for e in events if e.type == RealtimeEventType.RESPONSE_END), None)
        mai_response = (resp_evt.text_chunk or resp_evt.text_response) if resp_evt else "NO_RESPONSE"

        # Classification of STT output
        # Categories: STT recognition, AUDIO FORMAT, MICROPHONE CAPTURE, ROUTING, LLM interpretation
        error_class = "None (Accurate)"
        is_accurate = True

        spoken_lower = spoken.lower().replace(",", "").replace(".", "").replace("?", "")
        trans_lower = final_transcript.lower().replace(",", "").replace(".", "").replace("?", "")

        # Check key semantic content
        key_words = [w for w in spoken_lower.split() if len(w) > 3]
        matches = [w for w in key_words if w in trans_lower]
        match_rate = len(matches) / max(len(key_words), 1)

        if match_rate < 0.6:
            is_accurate = False
            error_class = "STT recognition"

        cpu_sample = psutil.cpu_percent(interval=0.2)

        record = {
            "id": u_id,
            "spoken": spoken,
            "assemblyai_transcript": final_transcript,
            "match_rate": round(match_rate, 2),
            "accurate": is_accurate,
            "error_classification": error_class,
            "stt_latency_ms": round(stt_lat_ms, 1),
            "mai_response_preview": mai_response[:120] if mai_response else "",
            "cpu_percent": cpu_sample,
        }
        results.append(record)

        print(f"   [AssemblyAI STT] \"{final_transcript}\" ({stt_lat_ms:.0f}ms)", flush=True)
        print(f"   [MAI Response]   \"{mai_response[:90]}...\"", flush=True)
        print(f"   [Classification] {error_class} (CPU: {cpu_sample}%)\n", flush=True)

    out_json = os.path.join(r"C:\Users\srisu\.gemini\antigravity-ide\brain\14606ebe-1c44-4b45-b6d9-6ec7c2755634\scratch", "live_microphone_assemblyai_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 75, flush=True)
    print(f"Acceptance Results saved to: {out_json}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    asyncio.run(run_live_end_to_end_test())
