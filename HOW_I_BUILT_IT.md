# How I Built This Model API

**Author:** Miles Robinson · **2026**

Project 5 is the **serve** half of ML. I already had a classifier story in Project 1; here the artifact is an HTTP service with tests, Docker, and CI.

---

## 1. Why FastAPI

I wanted OpenAPI `/docs`, Pydantic validation, and `TestClient` tests without a heavy framework. FastAPI + uvicorn is the default path most AI eng teams recognize.

---

## 2. Keep the model tiny on purpose

I embedded 24 SMS-like strings in `train_model.py` instead of re-downloading UCI. Reasons:

- Train in seconds on CPU
- No dataset license surprise in the API repo
- Forces the README to be honest: this is an **API demo**, not a SOTA spam paper

The joblib artifact is gitignored; CI and Docker train if missing.

---

## 3. Latency as a first-class signal

Middleware records:

- `X-Latency-Ms` response header
- JSONL lines in `results/request_log.jsonl` with a toy `cost_units` proxy

In interviews I can talk about SLOs and cost without pretending I ran a GPU cluster.

---

## 4. Tests before Docker

`tests/test_api.py` covers health, spam/ham predict, and empty-text validation. CI runs `train_model.py` then `pytest`. Docker is the packaging step, not the only way to verify.

---

## 5. Deploy notes without requiring deploy

README documents Railway/Render/Fly using the Dockerfile. If credentials aren't present, local uvicorn is enough for the portfolio bar.

---

## What I'd add next

- Auth (API key header) and rate limits
- Model versioning endpoint (`/model`)
- Swap the stub for Project 1's DistilBERT behind the same contract

*Miles Robinson · 2026*
