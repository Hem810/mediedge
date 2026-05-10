"""Smoke tests for MediEdge API.

Run: pytest tests/ -v

These tests verify the routes work end-to-end without hitting the Gemma API.
For testing AI inference, set GEMMA_API_KEY in your test env and run with
the --integration flag.
"""
import sys
from pathlib import Path

# Ensure parent dir is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from app import app
import asyncio
import database


# Initialise DB before tests
asyncio.run(database.init_db())

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "ollama" in data
    assert "host" in data["ollama"]
    assert "model" in data["ollama"]


def test_index_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "MediEdge" in response.text
    assert "Devanagari" in response.text  # Hindi font included


def test_dashboard_endpoint():
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "recent" in data
    assert isinstance(data["stats"]["total_assessments"], int)


def test_create_and_get_patient():
    # Create
    payload = {
        "name": "Test Patient",
        "age_months": 24,
        "sex": "F",
        "village": "Pilani",
        "phone": "9999999999",
        "worker_name": "Test Worker",
    }
    response = client.post("/api/patients", json=payload)
    assert response.status_code == 200
    patient = response.json()
    assert patient["name"] == "Test Patient"
    assert patient["age_months"] == 24
    assert "id" in patient

    # Get
    response = client.get(f"/api/patients/{patient['id']}")
    assert response.status_code == 200
    data = response.json()
    assert data["patient"]["name"] == "Test Patient"
    assert isinstance(data["assessments"], list)


def test_list_patients():
    response = client.get("/api/patients")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_nonexistent_patient_returns_404():
    response = client.get("/api/patients/nonexistent-uuid")
    assert response.status_code == 404


def test_list_assessments():
    response = client.get("/api/assessments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_transcribe_rejects_invalid_format():
    response = client.post(
        "/api/transcribe",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_kb_loads():
    """Verify the knowledge base loads without errors."""
    from services.kb_service import kb
    kb.load()
    assert len(kb._entries) > 0, "WHO IMCI knowledge base is empty"
    assert len(kb._drugs) > 0, "Drug formulary is empty"
    assert len(kb._referral_centres) > 0, "Referral centres are empty"


def test_kb_retrieval():
    """Verify BM25 retrieval returns relevant protocols."""
    from services.kb_service import kb
    results = kb.retrieve("child has fever and fast breathing", age_months=24)
    assert len(results) > 0
    # Should retrieve pneumonia or fever-related protocols
    titles = " ".join([r["title"].lower() for r in results])
    assert "pneumonia" in titles or "breathing" in titles or "fever" in titles


def test_kb_age_filtering():
    """Verify age filtering works — 7-month-old should not get adult-only protocols."""
    from services.kb_service import kb
    young_results = kb.retrieve("fever", age_months=7)
    for r in young_results:
        assert r.get("age_min_months", 0) <= 7
        assert r.get("age_max_months", 1200) >= 7


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
