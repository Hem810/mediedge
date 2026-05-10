"""All API routes - patients, assessments, audio transcription, image upload."""
from __future__ import annotations
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import database as db
from config import settings
from services.gemma_service import run_assessment
from services.stt_service import transcribe

router = APIRouter(prefix="/api")


# ── Models ────────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str
    age_months: int
    sex: str
    village: Optional[str] = None
    phone: Optional[str] = None
    worker_name: Optional[str] = None


class VitalsModel(BaseModel):
    temperature: Optional[float] = None
    heart_rate: Optional[int] = None
    resp_rate: Optional[int] = None
    spo2: Optional[int] = None
    weight: Optional[float] = None
    muac: Optional[float] = None


class AssessmentRequest(BaseModel):
    patient_id: str
    symptoms: str
    vitals: Optional[VitalsModel] = None
    image_path: Optional[str] = None
    worker_name: Optional[str] = None


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard():
    stats = await db.get_dashboard_stats()
    recent = await db.list_assessments(limit=8)
    return {"stats": stats, "recent": recent}


# ── Patients ──────────────────────────────────────────────────────────────────

@router.post("/patients")
async def create_patient(data: PatientCreate):
    return await db.create_patient(data.model_dump())


@router.get("/patients")
async def list_patients(q: str = "", limit: int = 30):
    return await db.search_patients(q, limit)


@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient = await db.get_patient(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    assessments = await db.list_assessments(patient_id=patient_id)
    return {"patient": patient, "assessments": assessments}


# ── Assessments ───────────────────────────────────────────────────────────────

@router.post("/assessments")
async def create_assessment(req: AssessmentRequest):
    patient = await db.get_patient(req.patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")

    try:
        result = await run_assessment(
            symptoms=req.symptoms,
            patient=patient,
            vitals=req.vitals.model_dump() if req.vitals else None,
            image_path=req.image_path,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    assessment_id = await db.create_assessment({
        "patient_id": req.patient_id,
        "worker_name": req.worker_name,
        "voice_transcript": req.symptoms,
        "vitals": req.vitals.model_dump() if req.vitals else None,
        "image_path": req.image_path,
        "differentials": result.get("differentials", []),
        "medications": result.get("medications", []),
        "referral": result.get("referral"),
        "soap_note": result.get("soap_note"),
        "plain_summary": result.get("plain_summary"),
        "plain_summary_hindi": result.get("plain_summary_hindi"),
        "followup_hindi": result.get("followup_hindi"),
        "followup": result.get("followup"),
        "overall_urgency": result.get("overall_urgency"),
        "inference_ms": result.get("inference_ms"),
    })

    return await db.get_assessment(assessment_id)


@router.get("/assessments")
async def list_assessments(limit: int = 50):
    return await db.list_assessments(limit)


@router.get("/assessments/{assessment_id}")
async def get_assessment(assessment_id: str):
    a = await db.get_assessment(assessment_id)
    if not a:
        raise HTTPException(404, "Assessment not found")
    return a


# ── Audio transcription ───────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Upload audio - returns Hindi transcript + English translation."""
    allowed = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".mp4"}
    suffix = Path(file.filename or "audio.webm").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported audio format: {suffix}")

    dest = settings.UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        result = await transcribe(str(dest))
    finally:
        dest.unlink(missing_ok=True)

    return result


# ── Image upload ──────────────────────────────────────────────────────────────

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported image format: {suffix}")

    filename = f"{uuid.uuid4()}{suffix}"
    dest = settings.UPLOAD_DIR / filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"image_path": str(dest), "filename": filename}
