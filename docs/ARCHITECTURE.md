# MediEdge — Architecture Deep Dive

This document is for engineers who want to understand how MediEdge works under the hood.

---

## System overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        BROWSER (ASHA worker)                     │
│  ┌────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐  │
│  │  Patient   │→│   Voice +    │→│   Vitals +   │→│  Result  │  │
│  │   form     │ │  transcript  │ │   image      │ │   page   │  │
│  └────────────┘ └──────────────┘ └──────────────┘ └──────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTPS
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI (uvicorn)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
│  │ /api/patients │ │ /api/transcribe│ │  /api/assessments       │  │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────────┘  │
│         │                │                    │                  │
│         ▼                ▼                    ▼                  │
│  ┌──────────┐    ┌────────────┐      ┌─────────────────┐         │
│  │  SQLite  │    │  Whisper   │      │  Gemma Service   │         │
│  │ (async)  │    │   STT      │      │   (orchestrator) │         │
│  └──────────┘    └────────────┘      └────┬─────────┬──┘         │
│                                            │         │            │
│                                            ▼         ▼            │
│                                  ┌─────────────┐  ┌──────────────┐ │
│                                  │ KB Service  │  │   Google AI  │ │
│                                  │   (BM25)    │  │   API        │ │
│                                  └──────┬──────┘  │  (Gemma 4)   │ │
│                                         │         └──────────────┘ │
│                                         ▼                          │
│                                  ┌──────────────┐                  │
│                                  │  data/*.json │                  │
│                                  │  WHO IMCI    │                  │
│                                  │  Drug list   │                  │
│                                  │  Referrals   │                  │
│                                  └──────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Request flow: an assessment

When a worker submits an assessment, here's what happens in 6–10 seconds:

### 1. Client → Server (~50 ms)

```js
POST /api/assessments
{
  "patient_id": "uuid-...",
  "symptoms": "Child has fever 4 days, cough, fast breathing...",
  "vitals": { "temperature": 39.2, "resp_rate": 56, "spo2": 91 },
  "image_path": "/app/uploads/abc.jpg",  // optional
  "worker_name": "Sushma"
}
```

### 2. Server: load patient + retrieve protocols (~10 ms)

```python
patient = await db.get_patient(patient_id)  # SQLite, ~5ms
context = kb.build_context(symptoms, age_months)  # BM25, ~5ms
```

The BM25 retrieval:
1. Filters the WHO IMCI corpus to age-appropriate entries (a 2-year-old gets entries with `age_min_months <= 24 <= age_max_months`).
2. Tokenises the symptom text.
3. Computes BM25 scores using k1=1.5, b=0.75 (standard parameters).
4. Returns top-5 protocols whose score > 0.

For Pari's case, this returns:
- "Cough and difficult breathing — child 2 months to 5 years"
- "Malaria assessment — child 2 months to 5 years"
- "General danger signs — all sick children"
- "Diarrhoea and dehydration..." (lower score, but in age band)
- "Anaemia and malaria..."

### 3. Server: build prompt + call Gemma (~5–8 sec)

The prompt has four sections:

```
[SYSTEM PROMPT]
You are MediEdge, a clinical decision support assistant for ASHA workers...
CRITICAL RULES:
1. Always prioritise patient safety...
2. Only NHM India Essential Medicines...
[etc.]

[USER PROMPT]
PATIENT: Pari, 2 yr 0 mo, F, Bissau
VITALS: Temp 39.2°C | RR 56/min | SpO2 91%
SYMPTOMS: Child has had high fever for 4 days, cough, very fast breathing.
          Not drinking milk, looks tired.
PROTOCOLS:
- [Pneumonia] Cough and difficult breathing: Count breaths for 1 minute.
  Fast breathing >=40 bpm (12-59 mo). Check chest indrawing...
- [Fever] Malaria assessment: ...
[3 more]
OUTPUT SCHEMA: { "differentials": [...], "medications": [...], ... }
```

This is sent to Google AI API with:
- `temperature: 0.1` (deterministic)
- `top_p: 0.95`
- `max_output_tokens: 2048`
- `response_mime_type: application/json` (forces JSON)

If an image is included, it's base64-encoded and sent as inline_data alongside the prompt.

### 4. Server: parse + persist (~20 ms)

```python
data = json.loads(response.text)
# Set defaults for missing fields
data.setdefault("differentials", [])
# Derive overall_urgency from top differential
data["overall_urgency"] = data["differentials"][0]["urgency"]

# Persist
assessment_id = await db.create_assessment({...})
return await db.get_assessment(assessment_id)
```

### 5. Client: render result

The frontend builds the HTML from the JSON: confidence ring gauges via SVG, urgency colour-coded sections, Hindi text in Noto Sans Devanagari, expandable differential cards.

**Total latency:** ~6 seconds for the first call (cold), ~3 seconds warm.

---

## Why these specific tech choices

### Why FastAPI?
- Native async/await — Whisper transcription, Gemma API call, and DB writes can be concurrent.
- Pydantic models give us request validation for free.
- Auto-generates OpenAPI docs at `/docs` — useful for the demo.
- Dead simple to deploy (no nginx/gunicorn complexity for a hackathon).

### Why SQLite, not Postgres?
- Single-tablet deployment is the realistic offline-first scenario.
- Zero ops overhead.
- aiosqlite gives us async without a separate DB server.
- One file → trivial to back up, sync, replicate.

### Why BM25, not vector embeddings?
1. The corpus is tiny (~12 protocols). Vector embeddings are overkill.
2. **Age-banding matters more than semantic similarity.** A query about a 7-month-old's diarrhoea must NEVER retrieve adult hypertension protocols, no matter how semantically close they look in some embedding space.
3. Zero ML deps means smaller container, faster cold start, fully offline-capable.
4. BM25 is interpretable — you can see exactly why a protocol was retrieved.

### Why Google AI API instead of self-hosted Gemma?
- For this hackathon submission, the API is free at 1,500 requests/day — enough for judges to demo.
- The architecture is intentionally one wrapper away from local Ollama. See `services/gemma_service.py` — replacing `genai.GenerativeModel` with `ollama.chat` is ~8 lines.
- This lets us demonstrate the system today, while the path to genuine offline operation (Common Service Centres running Gemma on commodity hardware) is documented.

### Why vanilla JS, not React?
- Single HTML file, ~700 lines, ~30 KB. Loads instantly on slow rural connections.
- No build step, no node_modules, no framework version churn.
- Anyone can read and modify it.

---

## Data layer

### Schema

```sql
CREATE TABLE patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age_months INTEGER NOT NULL,  -- months, not years
    sex TEXT NOT NULL,
    village TEXT,
    phone TEXT,
    worker_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE assessments (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    voice_transcript TEXT,         -- Hindi
    structured_symptoms TEXT,
    vitals_json TEXT,              -- JSON blob
    image_path TEXT,
    differentials_json TEXT,       -- JSON array
    medications_json TEXT,         -- JSON array
    referral_json TEXT,            -- JSON object
    soap_note TEXT,
    plain_summary TEXT,            -- English
    plain_summary_hindi TEXT,      -- Devanagari
    followup_hindi TEXT,
    overall_urgency TEXT,          -- 'high' | 'medium' | 'low' | 'observation'
    inference_ms REAL,
    status TEXT DEFAULT 'complete',
    created_at TEXT NOT NULL
);
```

### Index strategy

```sql
CREATE INDEX idx_assessments_patient ON assessments(patient_id);
CREATE INDEX idx_assessments_urgency ON assessments(overall_urgency);
CREATE INDEX idx_assessments_created ON assessments(created_at DESC);
```

The dashboard queries hit `idx_assessments_created` and `idx_assessments_urgency`. Patient timeline view hits `idx_assessments_patient`.

---

## Security considerations

For hackathon demo, we accept these tradeoffs. For production:

1. **Auth:** Add OAuth (Google/Microsoft) so each ASHA worker has their own caseload.
2. **PII:** Patient names and phone numbers are stored in plaintext in SQLite. For production, encrypt at rest (SQLCipher) and require auth to read.
3. **Audit log:** Add a separate audit table that logs every read of patient data with the worker's ID.
4. **API key handling:** Currently in `.env`. For production, use a secrets manager (AWS Secrets Manager, HashiCorp Vault).
5. **HTTPS:** Enforce via the deployment platform (Render/Railway/Fly all do this automatically).

---

## What we'd add next

1. **Offline mode via service worker** — cache the SPA, queue assessments locally, sync when online.
2. **Vector embeddings as a second retrieval pass** — for fuzzy symptom matching when BM25 misses.
3. **Fine-tune Gemma 4 with Unsloth** on Indian primary care vignettes → eligible for the Unsloth Special Track ($10K).
4. **Export to TFLite for Android** → eligible for the LiteRT Special Track ($10K).
5. **Local Ollama deployment guide** for the Common Service Centre rollout.
