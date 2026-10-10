import asyncio
import json
import websockets
from app.core.config import settings

async def test():
    key = settings.assemblyai_api_key
    url = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&speech_model=universal-3-6-pro"
    for i in range(3):
        print(f"Attempt {i+1} connecting...")
        ws = await websockets.connect(url, additional_headers={"Authorization": key})
        begin = await ws.recv()
        print(f"Attempt {i+1} connected:", json.loads(begin)["type"])
        await ws.send(json.dumps({"type": "Terminate"}))
        await ws.close()
        print(f"Attempt {i+1} closed. Waiting 0.5s...")
        await asyncio.sleep(0.5)
    print("ALL 3 ATTEMPTS SUCCESSFUL!")

if __name__ == "__main__":
    asyncio.run(test())
