"""API tests — train stub model if missing, then hit health + predict."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "spam_clf.joblib"


@pytest.fixture(scope="session", autouse=True)
def ensure_model() -> None:
    if not MODEL.exists():
        subprocess.check_call([sys.executable, str(ROOT / "train_model.py")], cwd=ROOT)


@pytest.fixture()
def client() -> TestClient:
    from app.main import app
    from app.model_loader import load_model

    load_model(force=True)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_spam(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "WINNER free iphone claim now http://bit.ly/x"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("ham", "spam")
    assert "spam" in body["probabilities"]
    assert body["latency_ms"] >= 0
    assert "X-Latency-Ms" in r.headers


def test_predict_ham(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "hey are we still on for lunch tomorrow"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("ham", "spam")


def test_predict_validation(client: TestClient) -> None:
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
