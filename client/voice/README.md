# P-MAI Reference Voice Client (P13)

This reference client provides a real-time conversational audio interface to the P-MAI voice backend (`/voice/realtime`).

## Architecture

```
Microphone
    ↓
AudioDevice (16kHz, 16-bit Mono PCM)
    ↓
PMAIVoiceClient (WebSocket)
    ↓
P-MAI Backend (/voice/realtime)
    ↓
Authoritative MAI Brain
    ↓
Streaming Audio Chunks
    ↓
Speaker Playback
```

## Setup & Dependencies

Install audio capture libraries if hardware microphone/speaker support is desired:
```bash
pip install pyaudio websockets
```
*Note: If `pyaudio` is not installed on the system, the client automatically falls back to synthetic PCM testing mode without failing.*

## Running the Voice Client

1. Ensure the P-MAI backend is running:
```bash
uvicorn app.main:app --reload --port 8000
```

2. Run the client script:
```python
import asyncio
from client.voice.client import PMAIVoiceClient

async def main():
    client = PMAIVoiceClient(server_ws_url="ws://localhost:8000/voice/realtime")
    await client.connect()
    print("Connected to P-MAI Voice Backend. Ready to converse.")

    # Speak / send turn
    chunks = client.device.generate_mock_speech_chunks(5)
    result = await client.run_turn(chunks)
    print("Response:", result["response_text"])
    print("Turn Metrics (ms):", result["metrics"])

    await client.close()

asyncio.run(main())
```
