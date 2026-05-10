"""STT Service - Hindi transcription via faster-whisper.

Uses faster-whisper (a CTranslate2 reimplementation of OpenAI Whisper) for
4x faster CPU inference and ~50% lower memory than the original library.
Same Whisper models, same accuracy.

Models are downloaded automatically from Hugging Face on first use and
cached at ~/.cache/huggingface/hub. After the first run everything works
fully offline.
"""
from __future__ import annotations
import asyncio

from config import settings

_whisper_model = None


def _get_whisper():
    """Load the Whisper model on first call. Subsequent calls reuse it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from e
        print(f"[STT] Loading faster-whisper '{settings.WHISPER_MODEL}' "
              f"(first run downloads ~500MB, subsequent runs are instant)...")
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",  # fastest on CPU; ~2x speedup vs float32
        )
        print("[STT] Whisper ready")
    return _whisper_model


def _transcribe_sync(audio_path: str) -> dict:
    """Run both Hindi transcription + English translation in one call."""
    model = _get_whisper()

    # Hindi transcription (keeps original language)
    segments, info = model.transcribe(
        audio_path,
        language="hi",
        task="transcribe",
        vad_filter=True,  # skip silent regions
    )
    transcript = " ".join(seg.text for seg in segments).strip()
    detected_lang = info.language

    # English translation for the LLM prompt
    translation = ""
    if detected_lang == "hi":
        segments, _ = model.transcribe(
            audio_path,
            language="hi",
            task="translate",
            vad_filter=True,
        )
        translation = " ".join(seg.text for seg in segments).strip()

    return {
        "transcript": transcript,
        "language": detected_lang,
        "translation": translation or transcript,
    }


async def transcribe(audio_path: str) -> dict:
    """Transcribe Hindi audio. Returns Hindi transcript + English translation.

    Returns:
        {
            "transcript": "<Hindi text in Devanagari>",
            "language": "hi",
            "translation": "<English translation>",
        }
    """
    # Run the synchronous Whisper call in a thread to not block the event loop
    return await asyncio.to_thread(_transcribe_sync, audio_path)
