"""MediEdge - application configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


class Settings:
    # Ollama config - local inference, no API key required
    # Install Ollama from https://ollama.com, then pull a model:
    #   ollama pull gemma4:e4b   (light - 4B params, runs on 8 GB RAM)
    #   ollama pull gemma4:31b   (heavy - 31B params, needs 32 GB RAM)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

    # Database (SQLite - perfect for offline-first, single-tablet deployment)
    DB_PATH: str = os.getenv("DB_PATH", str(BASE_DIR / "mediedge.db"))

    # Data files (knowledge base)
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    # App identity
    APP_TITLE: str = "MediEdge"
    APP_VERSION: str = "1.0.0"

    # Network
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Whisper STT model size
    # Options: tiny / base / small / medium / large
    # "small" gives the best balance of accuracy and speed for Hindi.
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
