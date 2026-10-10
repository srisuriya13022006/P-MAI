import asyncio
import json
import logging
import os
import time
import wave
import pyttsx3
import audioop
import websockets

from app.core.config import settings
from app.voice.stt.assemblyai_adapter import strip_wav_header_if_present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_v3")

def make_audio(text: str) -> bytes:
    engine = pyttsx3.init()
    os.makedirs("scratch", exist_ok=True)
    wav_path = "scratch/temp_v3_test.wav"
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    with wave.open(wav_path, "rb") as wf:
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
        rate = wf.getframerate()
        data_16k, _ = audioop.ratecv(data, 2, 1, rate, 16000, None)
        return data_16k

async def test_streaming(text: str):
    key = settings.assemblyai_api_key or os.environ.get("ASSEMBLYAI_API_KEY")
    url = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-6-pro"
    headers = {"Authorization": key}
    pcm = make_audio(text)
    
    t0 = time.perf_counter()
    async with websockets.connect(url, additional_headers=headers) as ws:
        connect_time = time.perf_counter() - t0
        begin_raw = await ws.recv()
        begin = json.loads(begin_raw)
        effective_model = begin.get("configuration", {}).get("model")
        logger.info(f"Connected in {connect_time*1000:.1f}ms. Effective model: {effective_model}")
        
        final_transcript = ""
        partials = []
        
        async def receive_loop():
            nonlocal final_transcript
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    mtype = data.get("type")
                    if mtype == "Turn":
                        transcript = data.get("transcript", "")
                        if data.get("end_of_turn"):
                            final_transcript = transcript
                            logger.info(f"FINAL TURN: '{transcript}'")
                        else:
                            partials.append(transcript)
                            logger.info(f"PARTIAL: '{transcript}'")
                    elif mtype == "Termination":
                        logger.info("Termination received from server.")
                        break
            except Exception as e:
                logger.debug(f"Receive loop ended: {e}")
        
        recv_task = asyncio.create_task(receive_loop())
        
        chunk_size = 3200  # 100ms
        stream_t0 = time.perf_counter()
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.08)  # Realtime pacing
        stream_duration = time.perf_counter() - stream_t0
        
        # Audio streaming done. Send termination to finalize turn.
        finalize_t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "Terminate"}))
        try:
            await asyncio.wait_for(recv_task, timeout=3.0)
        except asyncio.TimeoutError:
            pass
        finalize_time = time.perf_counter() - finalize_t0
        
        logger.info(f"Audio duration: {len(pcm)/(16000*2):.2f}s, Stream pacing: {stream_duration:.2f}s")
        logger.info(f"Finalize turnaround: {finalize_time*1000:.1f}ms")
        logger.info(f"Result: '{final_transcript}'")
        return {
            "text": text,
            "transcript": final_transcript,
            "effective_model": effective_model,
            "finalize_ms": finalize_time * 1000,
        }

if __name__ == "__main__":
    res = asyncio.run(test_streaming("What is your name?"))
    print("OUTPUT:", json.dumps(res, indent=2))
