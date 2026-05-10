"""Async SQLite persistence for patients, assessments, and sync queue."""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from typing import Optional

import aiosqlite

from config import settings


async def init_db() -> None:
    """Create tables on first run."""
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age_months INTEGER NOT NULL,
                sex TEXT NOT NULL,
                village TEXT,
                phone TEXT,
                worker_name TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                worker_name TEXT,
                voice_transcript TEXT,
                structured_symptoms TEXT,
                vitals_json TEXT,
                image_path TEXT,
                differentials_json TEXT,
                medications_json TEXT,
                referral_json TEXT,
                soap_note TEXT,
                plain_summary TEXT,
                plain_summary_hindi TEXT,
                followup_hindi TEXT,
                followup TEXT,
                overall_urgency TEXT,
                inference_ms REAL,
                status TEXT DEFAULT 'complete',
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_patient
                ON assessments(patient_id);
            CREATE INDEX IF NOT EXISTS idx_assessments_urgency
                ON assessments(overall_urgency);
            CREATE INDEX IF NOT EXISTS idx_assessments_created
                ON assessments(created_at DESC);
        """)
        await db.commit()


# ── Patients ──────────────────────────────────────────────────────────────────

async def create_patient(data: dict) -> dict:
    patient_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """INSERT INTO patients
               (id, name, age_months, sex, village, phone, worker_name, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (patient_id, data["name"], data["age_months"], data["sex"],
             data.get("village"), data.get("phone"), data.get("worker_name"), now),
        )
        await db.commit()
    return {**data, "id": patient_id, "created_at": now}


async def get_patient(patient_id: str) -> Optional[dict]:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM patients WHERE id=?", (patient_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def search_patients(query: str = "", limit: int = 30) -> list[dict]:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if query:
            async with db.execute(
                "SELECT * FROM patients WHERE name LIKE ? OR village LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


# ── Assessments ───────────────────────────────────────────────────────────────

async def create_assessment(data: dict) -> str:
    assessment_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """INSERT INTO assessments
               (id, patient_id, worker_name, voice_transcript, structured_symptoms,
                vitals_json, image_path, differentials_json, medications_json,
                referral_json, soap_note, plain_summary, plain_summary_hindi,
                followup_hindi, followup, overall_urgency, inference_ms, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                assessment_id,
                data["patient_id"],
                data.get("worker_name"),
                data.get("voice_transcript"),
                data.get("structured_symptoms"),
                json.dumps(data.get("vitals")) if data.get("vitals") else None,
                data.get("image_path"),
                json.dumps(data.get("differentials", [])),
                json.dumps(data.get("medications", [])),
                json.dumps(data.get("referral")) if data.get("referral") else None,
                data.get("soap_note"),
                data.get("plain_summary"),
                data.get("plain_summary_hindi"),
                data.get("followup_hindi"),
                data.get("followup"),
                data.get("overall_urgency"),
                data.get("inference_ms"),
                "complete",
                now,
            ),
        )
        await db.commit()
    return assessment_id


async def get_assessment(assessment_id: str) -> Optional[dict]:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT a.*, p.name as patient_name, p.age_months, p.sex,
                      p.village, p.phone
               FROM assessments a JOIN patients p ON a.patient_id=p.id
               WHERE a.id=?""",
            (assessment_id,),
        ) as cur:
            row = await cur.fetchone()
            return _deserialise(dict(row)) if row else None


async def list_assessments(limit: int = 50, patient_id: str | None = None) -> list[dict]:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if patient_id:
            sql = """SELECT a.*, p.name as patient_name, p.age_months, p.sex, p.village
                     FROM assessments a JOIN patients p ON a.patient_id=p.id
                     WHERE a.patient_id=? ORDER BY a.created_at DESC LIMIT ?"""
            params = (patient_id, limit)
        else:
            sql = """SELECT a.*, p.name as patient_name, p.age_months, p.sex, p.village
                     FROM assessments a JOIN patients p ON a.patient_id=p.id
                     ORDER BY a.created_at DESC LIMIT ?"""
            params = (limit,)

        async with db.execute(sql, params) as cur:
            return [_deserialise(dict(r)) for r in await cur.fetchall()]


async def get_dashboard_stats() -> dict:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        today = datetime.utcnow().date().isoformat()

        async def _count(sql: str, params=()) -> int:
            async with db.execute(sql, params) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

        return {
            "total_assessments": await _count("SELECT COUNT(*) FROM assessments"),
            "assessments_today": await _count(
                "SELECT COUNT(*) FROM assessments WHERE DATE(created_at)=?", (today,)
            ),
            "urgent_today": await _count(
                "SELECT COUNT(*) FROM assessments "
                "WHERE overall_urgency='high' AND DATE(created_at)=?", (today,)
            ),
            "total_patients": await _count("SELECT COUNT(*) FROM patients"),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deserialise(row: dict) -> dict:
    """Parse JSON columns into dicts/lists."""
    for field in ("vitals_json", "differentials_json", "medications_json", "referral_json"):
        val = row.pop(field, None)
        key = field.replace("_json", "")
        if val:
            try:
                row[key] = json.loads(val)
            except Exception:
                row[key] = None
        else:
            row[key] = None
    return row
