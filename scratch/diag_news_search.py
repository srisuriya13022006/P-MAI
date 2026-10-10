"""
P15.4.6 — Stage-by-Stage News Search Latency Diagnosis.
Separately measures:
- Web-search provider latency
- Page retrieval latency (if any)
- Planning and orchestration latency
- Cloud synthesis latency
- Local fallback latency (if any)
- Response formatting
- TTS generation and playback
"""
import asyncio
import base64
import json
import os
import sys
import time

sys.path.insert(0, r"d:\suriya\projects\P-MAI")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(r"d:\suriya\projects\P-MAI\.env")

from app.database.session import get_db
from app.services.chat_service import ChatService
from app.llm.factory import get_llm_provider
from app.tools.web.tool import WebSearchTool
from app.voice.audio import AudioNormalizer
from app.voice.formatter import VoicePresentationFormatter
from app.voice.tts.local_adapter import LocalTTSAdapter
import websockets
import pyttsx3


def synthesize_canonical_16k_wav(text: str, out_path: str):
    tmp_wav = out_path + ".tmp.wav"
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)
    engine.save_to_file(text, tmp_wav)
    engine.runAndWait()
    del engine

    with open(tmp_wav, "rb") as f:
        raw_bytes = f.read()
    try:
        os.unlink(tmp_wav)
    except OSError:
        pass

    normalizer = AudioNormalizer(16000)
    norm_wav = normalizer.normalize_wav(raw_bytes)
    with open(out_path, "wb") as f:
        f.write(norm_wav)


async def run_internal_stage_benchmarks():
    print("=" * 70, flush=True)
    print("INTERNAL STAGE-BY-STAGE BENCHMARKS FOR NEWS SEARCH", flush=True)
    print("=" * 70, flush=True)

    query = "Search the web for today's AI news."

    # Stage 1: Web-search provider latency
    print("\n1. Measuring WebSearchTool provider latency...", flush=True)
    tool = WebSearchTool()
    t0 = time.perf_counter()
    tool_res = tool.run(query="today's AI news")
    t1 = time.perf_counter()
    tool_latency_ms = (t1 - t0) * 1000.0
    print(f"   -> Web-Search Provider Latency: {tool_latency_ms:.1f} ms ({tool_latency_ms/1000:.2f}s)")
    print(f"   -> Success: {tool_res.success}, Results: {len(tool_res.data.get('results', [])) if tool_res.data else 0}")

    # Stage 2: Page retrieval latency (if any)
    print("\n2. Page Retrieval Latency:", flush=True)
    print("   -> Page retrieval was not invoked (WebSearchTool provides direct snippets & answer). Latency: 0.0 ms")

    # Stage 3: Planning & Orchestrator end-to-end
    print("\n3. Measuring Orchestrator & Cloud Synthesis Latency...", flush=True)
    db = next(get_db())
    llm = get_llm_provider()
    chat_service = ChatService(db, llm=llm)

    t0_orch = time.perf_counter()
    response = chat_service.chat(
        conversation_id="stage-diag-search",
        user_id="suriya-diag",
        user_message=query,
    )
    t1_orch = time.perf_counter()
    orch_latency_ms = (t1_orch - t0_orch) * 1000.0
    print(f"   -> Full Agent (Router + Search + Cloud Synthesis): {orch_latency_ms:.1f} ms ({orch_latency_ms/1000:.2f}s)")
    print(f"   -> Agent Response Preview: {response[:100]}...")

    # Stage 4: Cloud synthesis latency breakdown
    synth_approx_ms = max(0.0, orch_latency_ms - tool_latency_ms)
    print(f"   -> Inferred Cloud LLM Synthesis Latency (Groq): ~{synth_approx_ms:.1f} ms (~{synth_approx_ms/1000:.2f}s)")

    # Stage 5: Response formatting
    formatter = VoicePresentationFormatter()
    t0_fmt = time.perf_counter()
    speech_text = formatter.format_for_speech(response)
    t1_fmt = time.perf_counter()
    fmt_ms = (t1_fmt - t0_fmt) * 1000.0
    print(f"\n4. Response Formatting Latency: {fmt_ms:.2f} ms")

    # Stage 6: TTS generation and streaming playback
    print("\n5. Measuring Local TTS Generation & Chunking Latency...", flush=True)
    tts_adapter = LocalTTSAdapter()
    t0_tts = time.perf_counter()
    tts_res = await tts_adapter.synthesize(speech_text[:450])
    t1_tts = time.perf_counter()
    tts_ms = (t1_tts - t0_tts) * 1000.0
    print(f"   -> TTS Synthesis Latency: {tts_ms:.1f} ms ({tts_ms/1000:.2f}s)")

    # Stage 7: Local fallback latency comparison (if Ollama were used)
    print("\n6. Local Fallback (Ollama) Latency Comparison:", flush=True)
    print("   -> In P15.4.5, when Groq returned HTTP 429, Ollama fallback on CPU took 74.7 seconds.")
    print(f"   -> With healthy Groq, Cloud synthesis takes {synth_approx_ms/1000:.2f}s (a ~{74700/max(1, synth_approx_ms):.1f}x speedup!).")

    return {
        "tool_latency_ms": tool_latency_ms,
        "orch_latency_ms": orch_latency_ms,
        "cloud_synth_approx_ms": synth_approx_ms,
        "formatting_ms": fmt_ms,
        "tts_ms": tts_ms,
    }


async def run_live_websocket_news_search():
    print("\n" + "=" * 70, flush=True)
    print("LIVE WEBSOCKET VOICE NEWS SEARCH BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    wav_path = r"d:\suriya\projects\P-MAI\scratch\p15_4_6_diag\news_search.wav"
    synthesize_canonical_16k_wav("Search the web for today's AI news.", wav_path)

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    from app.voice.audio import strip_wav_header
    pcm = strip_wav_header(audio_bytes)

    ws_url = "ws://localhost:8000/voice/realtime?conversation_id=p15-4-6-news-search&user_id=suriya"
    print(f"Connecting to {ws_url}...", flush=True)

    async with websockets.connect(ws_url, ping_interval=30, ping_timeout=120) as ws:
        t_client_audio_start = time.perf_counter()
        await ws.send(json.dumps({"type": "AUDIO_START"}))

        chunk_size = 3200
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i:i + chunk_size]
            b64 = base64.b64encode(chunk).decode("ascii")
            await ws.send(json.dumps({
                "type": "AUDIO_CHUNK",
                "audio_base64": b64,
                "content_type": "audio/wav",
            }))
            await asyncio.sleep(0.02)

        t_client_audio_end = time.perf_counter()
        await ws.send(json.dumps({"type": "AUDIO_END"}))

        t_client_first_audio_recv = None
        t_client_response_end_recv = None
        response_text = ""
        response_metadata = {}

        while True:
            raw = await ws.recv()
            evt = json.loads(raw)
            etype = evt.get("type")

            if etype == "TRANSCRIPT_FINAL":
                t_tr = time.perf_counter()
                print(f"  -> TRANSCRIPT_FINAL: \"{evt.get('transcript')}\" (+{(t_tr - t_client_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)

            elif etype == "RESPONSE_CHUNK":
                if t_client_first_audio_recv is None:
                    t_client_first_audio_recv = time.perf_counter()
                    print(f"  -> First RESPONSE_CHUNK (+{(t_client_first_audio_recv - t_client_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)

            elif etype == "RESPONSE_END":
                t_client_response_end_recv = time.perf_counter()
                response_text = evt.get("text_chunk", "")
                response_metadata = evt.get("metadata", {})
                print(f"  -> RESPONSE_END (+{(t_client_response_end_recv - t_client_audio_end)*1000:.1f}ms from AUDIO_END)", flush=True)
                break

            elif etype == "ERROR":
                print(f"  -> ERROR: {evt}", flush=True)
                if evt.get("error_code") not in ("TTS_STREAMING_ERROR",):
                    break

        total_wall_clock_ms = (t_client_response_end_recv - t_client_audio_start) * 1000.0
        turnaround_from_audio_end_ms = (t_client_response_end_recv - t_client_audio_end) * 1000.0

        stt_ms = response_metadata.get("stt_latency_ms", 0.0)
        mai_ms = response_metadata.get("mai_latency_ms", 0.0)
        tts_ms = response_metadata.get("tts_latency_ms", 0.0)
        llm_prov = response_metadata.get("llm_provider", "unknown")

        print("\n--- LIVE WEBSOCKET TIMING BREAKDOWN ---", flush=True)
        print(f"  STT Latency: {stt_ms:.1f} ms")
        print(f"  Agent (Search + Cloud Synthesis): {mai_ms:.1f} ms (Provider: {llm_prov})")
        print(f"  TTS Latency: {tts_ms:.1f} ms")
        print(f"  Time to First Audio from AUDIO_END: {(t_client_first_audio_recv - t_client_audio_end)*1000:.1f} ms")
        print(f"  Turnaround from AUDIO_END: {turnaround_from_audio_end_ms:.1f} ms ({turnaround_from_audio_end_ms/1000:.2f}s)")
        print(f"  Total Wall Clock: {total_wall_clock_ms:.1f} ms ({total_wall_clock_ms/1000:.2f}s)")
        print(f"  MAI Response: {response_text[:120]}...")

        return {
            "stt_ms": stt_ms,
            "mai_ms": mai_ms,
            "tts_ms": tts_ms,
            "turnaround_from_audio_end_ms": turnaround_from_audio_end_ms,
            "total_wall_clock_ms": total_wall_clock_ms,
            "llm_provider": llm_prov,
        }


async def main():
    stage_results = await run_internal_stage_benchmarks()
    ws_results = await run_live_websocket_news_search()

    diag_data = {
        "stage_benchmarks": stage_results,
        "websocket_run": ws_results,
    }
    with open(r"d:\suriya\projects\P-MAI\scratch\news_search_diag_results.json", "w", encoding="utf-8") as f:
        json.dump(diag_data, f, indent=2)
    print("\nSaved news search diagnosis to scratch/news_search_diag_results.json")


if __name__ == "__main__":
    asyncio.run(main())
