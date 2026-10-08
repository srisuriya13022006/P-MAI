import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_memory import router as memory_router
from app.api.routes_voice import router as voice_router
from app.core.config import settings


app = FastAPI(title=settings.app_name)

app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(voice_router)

# Mount voice web client static assets if available
web_client_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "client", "web")
if os.path.exists(web_client_dir):
    app.mount("/static/voice", StaticFiles(directory=web_client_dir), name="voice_static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }