"""
Gemma Service - calls local Ollama for clinical reasoning.

NO API KEY REQUIRED. Runs Gemma 4 entirely on your machine.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull the model:
       ollama pull gemma4:e4b     # 4B params - laptops, 8GB RAM (default)
       ollama pull gemma4:31b     # 31B params - best quality, needs 32GB RAM
  3. Run the model (Ollama starts a local server on port 11434)
"""
from __future__ import annotations
import json
import time
import base64
import asyncio
from pathlib import Path
from typing import Optional

import ollama

from config import settings
from services.kb_service import kb


def _make_client() -> ollama.Client:
    """Create an Ollama client pointed at the configured host."""
    return ollama.Client(host=settings.OLLAMA_HOST)


def check_ollama_health() -> dict:
    """
    Verify Ollama is running and the configured model is available.
    Returns: {"ok": bool, "message": str, "models": list[str]}
    """
    try:
        client = _make_client()
        result = client.list()
        # ollama 0.4.x returns objects with .model attribute, older returns dicts with 'name'
        models = []
        for m in (result.models if hasattr(result, "models") else result.get("models", [])):
            name = getattr(m, "model", None) or m.get("model") or m.get("name")
            if name:
                models.append(name)

        # Match by prefix - settings.OLLAMA_MODEL might be "gemma3:4b" and the listed
        # model might be "gemma3:4b" or "gemma3:4b-it-q4_K_M"
        target = settings.OLLAMA_MODEL
        has_model = any(m.startswith(target) or target.startswith(m.split(":")[0]) for m in models)

        if not has_model:
            return {
                "ok": False,
                "message": f"Model '{target}' not found. Pull it with: ollama pull {target}",
                "models": models,
            }
        return {"ok": True, "message": f"Ready ({target})", "models": models}
    except Exception as e:
        return {
            "ok": False,
            "message": f"Cannot reach Ollama at {settings.OLLAMA_HOST}. "
                       f"Is it running? Install from https://ollama.com (error: {e})",
            "models": [],
        }


SYSTEM_PROMPT = """You are MediEdge, a clinical decision support assistant for ASHA (Accredited Social Health Activist) workers in rural India. You strictly follow WHO IMCI guidelines and NHM India Essential Medicines protocols.

CRITICAL RULES:
1. Always prioritise patient safety - when in doubt, recommend referral
2. Only recommend medicines from the NHM India Essential Medicines List
3. Provide weight-based dosing for children under 5
4. Use ICD-10 codes for diagnoses
5. Output MUST be valid JSON matching the schema exactly
6. Plain language summaries must be in simple Hindi (Devanagari script)
7. Never invent drug names or dosages - use only what you know is safe
8. For any general danger sign (lethargy, convulsions, inability to drink, vomits everything), recommend urgent referral

You are assisting a minimally-trained health worker in a village with no doctor. Your output will directly guide their actions on a sick patient.

You MUST respond with ONLY a valid JSON object - no markdown fences, no preamble, no explanation. Just JSON."""


OUTPUT_SCHEMA = """{
  "differentials": [
    {
      "icd_code": "string",
      "name": "string (English)",
      "name_hindi": "string (Hindi Devanagari)",
      "confidence": 0.0 to 1.0,
      "supporting_findings": ["string"],
      "against_findings": ["string"],
      "urgency": "high | medium | low | observation",
      "who_imci_category": "string or null",
      "treatment_hindi": "1-2 sentence treatment summary in Hindi",
      "treatment": "1-2 sentence treatment summary in English"
    }
  ],
  "medications": [
    {
      "generic_name": "string",
      "generic_name_hindi": "string",
      "dose": "exact amount",
      "dose_hindi": "in Hindi",
      "frequency": "OD | BD | TDS | QDS | SOS",
      "duration": "e.g. 5 days",
      "route": "oral | topical | IM | IV | rectal",
      "warning": "string or null",
      "in_nhm_formulary": true or false
    }
  ],
  "referral": {
    "needed": true or false,
    "urgency": "high | medium | low",
    "reason": "string (English)",
    "reason_hindi": "string (Hindi)"
  },
  "soap_note": "professional SOAP format clinical note",
  "plain_summary": "simple English summary for health worker",
  "plain_summary_hindi": "simple Hindi summary (max 3 sentences, Devanagari)",
  "followup_hindi": "follow-up advice in Hindi",
  "followup": "follow-up advice in English"
}"""


async def run_assessment(
    symptoms: str,
    patient: dict,
    vitals: Optional[dict] = None,
    image_path: Optional[str] = None,
) -> dict:
    """Run clinical assessment using local Gemma via Ollama. Returns parsed dict."""
    start = time.perf_counter()

    age_months = patient.get("age_months", 60)
    protocols = kb.build_context(symptoms, age_months)
    prompt = _build_prompt(symptoms, patient, vitals, protocols)

    try:
        result = await asyncio.to_thread(_call_ollama, prompt, image_path)
    except Exception as e:
        msg = str(e)
        if "connection" in msg.lower() or "refused" in msg.lower():
            raise RuntimeError(
                f"Cannot reach Ollama at {settings.OLLAMA_HOST}. "
                f"Make sure Ollama is running (install from https://ollama.com)."
            ) from e
        if "not found" in msg.lower() or "no such" in msg.lower():
            raise RuntimeError(
                f"Model '{settings.OLLAMA_MODEL}' not pulled. "
                f"Run: ollama pull {settings.OLLAMA_MODEL}"
            ) from e
        raise RuntimeError(f"Ollama inference failed: {e}") from e

    elapsed_ms = (time.perf_counter() - start) * 1000

    assessment = _parse_output(result)
    assessment["inference_ms"] = round(elapsed_ms, 1)
    return assessment


def _call_ollama(prompt: str, image_path: Optional[str]) -> str:
    """Synchronous Ollama call - run via asyncio.to_thread."""
    client = _make_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Multimodal: attach image as base64 to the user message
    if image_path and Path(image_path).exists():
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        # Ollama expects images as a list of base64 strings on the message
        messages[-1]["images"] = [b64]
        messages[-1]["content"] = (
            "CLINICAL IMAGE ATTACHED. Please analyse the image for relevant findings.\n\n"
            + messages[-1]["content"]
        )

    response = client.chat(
        model=settings.OLLAMA_MODEL,
        messages=messages,
        format="json",   # Forces valid JSON output (Ollama feature)
        options={
            "temperature": 0.1,
            "top_p": 0.95,
            "num_predict": 2048,
        },
    )

    # ollama 0.4.x returns object with .message.content; older returned dict
    msg = response.message if hasattr(response, "message") else response.get("message", {})
    content = getattr(msg, "content", None) or msg.get("content", "")
    return content


def _build_prompt(symptoms: str, patient: dict, vitals: Optional[dict], protocols: str) -> str:
    age_months = patient.get("age_months", 0)
    yr, mo = age_months // 12, age_months % 12
    age_str = f"{yr} yr {mo} mo" if age_months else "unknown"

    vitals_str = "Not measured"
    if vitals:
        parts = []
        if vitals.get("temperature"): parts.append(f"Temp: {vitals['temperature']}C")
        if vitals.get("heart_rate"):  parts.append(f"HR: {vitals['heart_rate']} bpm")
        if vitals.get("resp_rate"):   parts.append(f"RR: {vitals['resp_rate']}/min")
        if vitals.get("spo2"):        parts.append(f"SpO2: {vitals['spo2']}%")
        if vitals.get("weight"):      parts.append(f"Weight: {vitals['weight']} kg")
        if vitals.get("muac"):        parts.append(f"MUAC: {vitals['muac']} cm")
        vitals_str = " | ".join(parts) or "Not measured"

    return f"""PATIENT ASSESSMENT REQUEST

Patient: {patient.get('name', 'Unknown')}
Age: {age_str} | Sex: {patient.get('sex', '?')} | Village: {patient.get('village', 'Unknown')}

Vitals: {vitals_str}

Presenting symptoms (transcribed from health worker's voice in Hindi):
{symptoms}

Relevant WHO IMCI protocols retrieved for this age band:
{protocols}

Required: Use only NHM India Essential Medicines for any drug recommendations.

OUTPUT SCHEMA (respond with ONLY valid JSON, no markdown, no preamble):
{OUTPUT_SCHEMA}"""


def _parse_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip().rstrip("`").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return _safety_fallback()
        else:
            return _safety_fallback()

    data.setdefault("differentials", [])
    data.setdefault("medications", [])
    data.setdefault("referral", {"needed": False})
    data.setdefault("soap_note", "")
    data.setdefault("plain_summary", "")
    data.setdefault("plain_summary_hindi", "")
    data.setdefault("followup_hindi", "")
    data.setdefault("followup", "")

    if data["differentials"]:
        data["overall_urgency"] = data["differentials"][0].get("urgency", "observation")
    else:
        data["overall_urgency"] = "observation"

    return data


def _safety_fallback() -> dict:
    return {
        "differentials": [{
            "icd_code": "Z00.0",
            "name": "Assessment incomplete - refer to PHC",
            "name_hindi": "जांच अधूरी - PHC भेजें",
            "confidence": 1.0,
            "supporting_findings": ["System could not complete analysis"],
            "against_findings": [],
            "urgency": "medium",
            "who_imci_category": None,
            "treatment_hindi": "PHC या CHC में तुरंत दिखाएं।",
        }],
        "medications": [],
        "referral": {
            "needed": True, "urgency": "medium",
            "reason": "Could not complete assessment",
            "reason_hindi": "जांच पूरी नहीं हो सकी",
        },
        "soap_note": "Assessment could not be completed. Patient should be referred.",
        "plain_summary": "Analysis failed. Refer patient to nearest PHC.",
        "plain_summary_hindi": "जांच नहीं हो सकी। मरीज को PHC भेजें।",
        "followup_hindi": "नजदीकी PHC में जल्द दिखाएं।",
        "followup": "Show patient at the nearest PHC soon.",
        "overall_urgency": "medium",
    }
