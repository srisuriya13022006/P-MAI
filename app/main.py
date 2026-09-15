from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.core.config import settings


app = FastAPI(title=settings.app_name)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }