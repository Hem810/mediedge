<div align="center">

# 🏥 MediEdge

### Offline AI clinical decision support for India's 900,000 ASHA workers

**Powered by Gemma 4, running locally via Ollama. No API key. No cloud. No internet.**

[![Gemma 4 Good Hackathon](https://img.shields.io/badge/Gemma_4_Good_Hackathon-2026-1D9E75?style=for-the-badge)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Health & Sciences](https://img.shields.io/badge/Track-Health_%26_Sciences-D95F3B?style=for-the-badge)](#-hackathon-submission)
[![Ollama Special Tech](https://img.shields.io/badge/Track-Ollama_Special_Tech-0A6E5C?style=for-the-badge)](#-hackathon-submission)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma_4-000000?style=flat-square)](https://ollama.com)

🎬 **[Watch the 3-minute demo](https://youtu.be/BhsvpntCbnE)** &nbsp;·&nbsp; 📄 **[Read the writeup](docs/KAGGLE_WRITEUP.md)** &nbsp;·&nbsp; 🏗️ **[Architecture](docs/ARCHITECTURE.md)** &nbsp;·&nbsp;

</div>

---

## The 10:30 PM Problem

It is 10:30 PM in Bissau, a village in Jhunjhunu district, Rajasthan. A one year old boy named Rahul has had a fever for three days. His mother walks 400 metres to the small concrete house where Sushma , the village's ASHA (Accredited Social Health Activist) worker, lives.

She has a Class 12 education, three months of community health training, and a paper booklet from 2018. She has no doctor on call. The nearest Community Health Centre is 14.5 km away. The cell tower three villages over has been out since the morning rain.

She has to decide, by herself, in the next thirty seconds, whether Rahul needs to travel that 14.5 km tonight.

This decision plays out **roughly 13 million times every week** across rural India. Get it wrong by under referring and a child dies of pneumonia. Get it wrong by over referring and a daily wage family loses an entire week's income.

**Until now.**

---


## What it does

MediEdge is a web-based clinical decision support assistant. The ASHA worker opens it on any browser — phone, tablet, or laptop — and walks through a four-step assessment:

1. **Patient details** — name, age in months (critical for WHO IMCI age-banding), sex, village
2. **Voice symptom recording in Hindi** — `faster-whisper` transcribes locally with English translation for the model
3. **Vitals + optional clinical photo** — abnormal values flag in red automatically
4. **Gemma 4 analyses** — retrieves relevant WHO IMCI protocols via BM25, calls local Ollama, returns structured JSON

In **60–70 seconds**, she sees:

- Ranked **differential diagnosis** with confidence scores, ICD-10 codes, supporting and against findings, mapped to WHO IMCI categories
- **Medications** drawn exclusively from the NHM India Essential Medicines list, with weight-based paediatric dosing in **both Hindi and English**
- **Referral decision** with the nearest facility, phone number, urgency, and a one-line reason
- **Plain-language Hindi summary** she can read aloud to the patient's family — three sentences in Devanagari, with English italicised below

Every Hindi line in the interface has English directly underneath. The same is true of every model output. This isn't decoration — it's how an ASHA worker can verify her own understanding, and how a supervising medical officer can audit the record later.

---

## Quick start (5 minutes)

### Prerequisites

- Python 3.11 or later

### Step 1 — Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer from https://ollama.com/download
```

### Step 2 — Pull a Gemma 4 model

```bash
ollama pull gemma4:e4b
```

### Step 3 — Run MediEdge

```bash
git clone https://github.com/Hem810/mediedge
cd mediedge
pip install -r requirements.txt
python app.py
```

Open <http://localhost:8000>.

That's it. **No API keys. No accounts. No data leaves your machine.**

The first time you record a Hindi clip, `faster-whisper` will download the small model (~500 MB) — every subsequent recording runs from the local cache.

---

## How it works

```
[Voice]──faster-whisper (local)──┐
[Image]──────────────────────────┤
[Vitals]─────────────────────────┼──┐
                                    ▼
                          ┌────────────────────────┐
                          │   Gemma 4 · Ollama     │
                          │   running on YOUR      │
                          │   machine              │
                          └─────┬─────────────────┬┘
                                │                 │
        ┌───────────────────────┘                 └───────────────┐
[WHO IMCI knowledge base]                              [NHM Drug Formulary]
   BM25 retrieval, age-banded                                11 medicines
        │                                                        │
        └────────────────┬───────────────────────────────────────┘
                         ▼
              ┌──────────────────────┐
              │   JSON Assessment    │
              │   ────────────────   │
              │   Differentials      │
              │   Medications        │
              │   Referral decision  │
              │   SOAP note          │
              │   Bilingual summary  │
              └──────────┬───────────┘
                         ▼
                  [SQLite + Web UI]
```

Every component runs locally. The architecture has no cloud dependency by design — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full deep-dive.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| **Reasoning** | Gemma 4 (`gemma4:e4b` / `gemma4:31b`) via [Ollama](https://ollama.com) | Local inference, multimodal, JSON-mode output, open weights |
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Hindi transcription on CPU, ~4× faster than `openai-whisper`, no PyTorch dependency |
| **Backend** | FastAPI + Uvicorn | Async, typed, tested, auto-generated OpenAPI docs |
| **Database** | SQLite via aiosqlite | Single file, zero ops, perfect for offline-first single-tablet deployment |
| **Knowledge retrieval** | BM25 over JSON corpus | Age-banded, interpretable, no ML dependencies, ideal for a small clinical KB |
| **Frontend** | Vanilla JS + DM Serif Display + Noto Sans Devanagari | One HTML file, ~30 KB, loads instantly, no build step, no framework churn |

---

## Project structure

```
mediedge/
├── app.py                       # FastAPI entry point
├── config.py                    # Settings (.env loader)
├── database.py                  # Async SQLite (aiosqlite)
│
├── services/
│   ├── gemma_service.py         # Gemma 4 inference via local Ollama
│   ├── kb_service.py            # WHO IMCI BM25 retrieval
│   └── stt_service.py           # faster-whisper Hindi STT
│
├── routers/
│   └── api.py                   # REST endpoints
│
├── templates/
│   └── index.html               # Single-page app frontend (bilingual)
│
├── data/
│   ├── who_imci.json            # 12 WHO IMCI protocols
│   ├── drug_formulary.json      # 11 NHM essential medicines
│   └── referral_centres.json    # Rajasthan PHC/CHC directory
│
├── docs/
│   ├── KAGGLE_WRITEUP.md        # Hackathon writeup (~1,290 words)
│   └── ARCHITECTURE.md          # Engineering deep-dive 
│
├── tests/
│   └── test_smoke.py            # 11 API smoke tests
│
├── Dockerfile                   # Production container
├── docker-compose.yml           # Bundles Ollama + web app
├── requirements.txt
└── .env.example
```

---

## Why local-first matters

This isn't a stylistic choice — it's the only architecture that works for the actual user.

**ASHA workers don't have reliable internet.** A village in Jhunjhunu district can lose connectivity for days at a time. An app that requires API calls is an app they cannot use when they need it most.

**Patient data should not leave the village.** Health records of women and children in rural India have a long, troubled history of being mishandled by external systems. Sending vitals and symptoms to a foreign cloud is a real privacy harm. Local inference means the data never leaves the tablet.

**Per-query API costs don't scale to 900,000 workers.** Even at $0.001 per call, 13 million weekly consultations is over $13,000 per week. No NGO has that budget. Local inference is ₹0 per query after the model download.

**The model is yours.** No vendor can deprecate it. No terms-of-service change can disrupt the village clinic. No rate limit can fail at 10:30 PM when a child has a fever.

---

## Testing

```bash
pytest tests/ -v
```

Eleven smoke tests covering API endpoints, knowledge base loading, BM25 retrieval, and age filtering. CI runs them via GitHub Actions on every push.

---

## Docker (full stack with one command)

```bash
# First time: pull the model
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:e4b

# Then: run everything
docker compose up
```

The `docker-compose.yml` runs both an Ollama service and the MediEdge web app. Models persist in a Docker volume — subsequent runs are instant.

---

## Hackathon submission

This project was built for the [**Gemma 4 Good Hackathon**](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (April–May 2026), targeting:

- 🩺 **Health & Sciences** — clinical decision support for underserved communities
- ⚡ **Ollama Special Tech** — fully local Gemma inference, no cloud
- 🏆 **Main Track** — addressing a real-world problem with Gemma 4

### For judges — evaluate in 5 minutes

1. **Watch the 3-minute demo video** (link at the top of this README)
2. **Read the writeup**: [`docs/KAGGLE_WRITEUP.md`](docs/KAGGLE_WRITEUP.md)
3. **Skim the architecture**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
4. **Or run it yourself** — the 5-minute Quick Start above gives you a working installation

### Things worth a closer look

- [`services/gemma_service.py`](services/gemma_service.py) — local Ollama integration with structured JSON output
- [`services/kb_service.py`](services/kb_service.py) — age-banded BM25 retrieval over WHO IMCI
- [`templates/index.html`](templates/index.html) — bilingual UI: every Hindi string has English directly below
- [`data/who_imci.json`](data/who_imci.json) — the clinical protocol corpus
- The `format: "json"` pattern in `_call_ollama()` for reliable structured output from a local model

---

## Roadmap

This is v1.0. The architecture supports several natural extensions:

- **Fine-tune Gemma 4 with Unsloth** on Indian primary-care vignettes (eligible for the Unsloth Special Tech track) — the fine-tuned model packages cleanly as a custom Ollama Modelfile
- **Export to TFLite** for true on-device inference on Android (eligible for the LiteRT track) — letting the app run with no server at all
- **Expand the knowledge base** to all 47 WHO IMCI conditions (currently 12)
- **Pilot deployment** in Jhunjhunu district with partner ASHA training programmes
- **Common Service Centre rollout** — India has 250,000 CSCs in panchayats, each capable of running a single Ollama instance for an entire taluka

---

## ⚠️ Important medical disclaimer

MediEdge is a **clinical decision support tool** intended to assist trained health workers. It is **not** a replacement for medical professionals and should **not** be the sole basis for medical decisions. The system always recommends referral when uncertain, and outputs require trained human judgement before acting. Always seek qualified medical care.

---

## License

[MIT](LICENSE) — free for any use, including commercial.

The WHO IMCI protocols are public-domain WHO publications. The NHM Essential Medicines List is published by the Government of India.

---

<div align="center">

### *The characters are fictional. The problem isn't.*

**MediEdge — frontier AI for the last health mile.**

Built with care for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) · 2026

</div>
