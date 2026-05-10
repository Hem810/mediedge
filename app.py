"""
MediEdge Web - FastAPI application entry point.

NO API KEY REQUIRED. Runs Gemma locally via Ollama.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model: ollama pull gemma3:4b   (or gemma3:12b / gemma3:27b)
    3. pip install -r requirements.txt
    4. python app.py

The app will check Ollama on startup and warn if it cannot reach it.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database
from config import settings
from routers.api import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[MediEdge] Initialising database at {settings.DB_PATH}")
    await database.init_db()

    from services.kb_service import kb
    kb.load()
    print(f"[MediEdge] Knowledge base loaded - {len(kb._entries)} IMCI entries, {len(kb._drugs)} drugs")

    print(f"[MediEdge] Server ready at http://{settings.HOST}:{settings.PORT}")
    print(f"[MediEdge] Ollama host: {settings.OLLAMA_HOST}")
    print(f"[MediEdge] Model: {settings.OLLAMA_MODEL}")

    # Check Ollama on startup
    from services.gemma_service import check_ollama_health
    health = check_ollama_health()
    if health["ok"]:
        print(f"[MediEdge] Ollama: {health['message']}")
    else:
        print(f"[MediEdge] WARNING: {health['message']}")
        if health.get("models"):
            print(f"[MediEdge]   Available models: {', '.join(health['models'])}")
        else:
            print("[MediEdge]   1. Install Ollama: https://ollama.com/download")
            print(f"[MediEdge]   2. Pull the model: ollama pull {settings.OLLAMA_MODEL}")

    yield


app = FastAPI(
    title="MediEdge",
    description="Offline AI health assistant for rural ASHA workers - powered by local Gemma via Ollama",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    from services.gemma_service import check_ollama_health
    ollama = check_ollama_health()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "ollama": {
            "host": settings.OLLAMA_HOST,
            "model": settings.OLLAMA_MODEL,
            "ready": ollama["ok"],
            "message": ollama["message"],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
