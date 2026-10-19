# Model API (FastAPI)

**Author:** Miles Robinson · **Year:** 2026  
**Portfolio:** Project 5 of an AI engineering series  
**Stack:** FastAPI · scikit-learn (TF-IDF + LogReg) · uvicorn · Docker · GitHub Actions

A **deployable model API** that serves a tiny spam/ham text classifier. Local `uvicorn` + pytest green path, Dockerfile / docker-compose, CI workflow, and request **latency / cost-style logging**. Free-tier deploy notes (Railway / Render / Fly) included — live deploy is optional if credentials are missing.

Also see [SMS Spam vs Ham Classifier](https://github.com/1ipkiy/sms-spam-classifier) · [RAG Q&A with Citations](https://github.com/1ipkiy/rag-qa-with-citations) · [LLM Eval Harness](https://github.com/1ipkiy/llm-eval-harness) · [Tool-Using Agent](https://github.com/1ipkiy/tool-using-agent).

---

## Problem statement

Training a model is half the job. The other half is **serving** it:

1. Stable HTTP contract (`/health`, `/predict`)
2. Latency visibility on every request
3. Tests + CI so refactors don't silently break inference
4. A container path for free-tier hosts

This repo is the MLOps-flavored sibling of Project 1's classifier — a stub model kept tiny so the **API engineering** stays the focus.

---

## Dataset / model

| Field | Value |
|-------|--------|
| Task | Binary SMS-like text: `ham` vs `spam` |
| Data | 24 hand-written stub messages in `train_model.py` (not the full UCI corpus) |
| Split | 75% train / 25% test, stratified, seed=42 |
| Model | `TfidfVectorizer` (1–2 grams) + `LogisticRegression(class_weight="balanced")` |
| Artifact | `model/spam_clf.joblib` (gitignored — regenerate with `train_model.py`) |

---

## Approach

| Piece | Choice | Why |
|-------|--------|-----|
| Framework | **FastAPI** | Typed `/docs`, easy TestClient tests |
| Model | sklearn TF-IDF + LogReg | CPU-only, seconds to train, no GPU |
| Logging | `X-Latency-Ms` header + `results/request_log.jsonl` | Latency + toy `cost_units` proxy |
| Container | Dockerfile + optional compose | Same entrypoint for local / cloud |
| CI | `.github/workflows/ci.yml` | Train → pytest on every push/PR |

Pipeline: `train_model.py` → `uvicorn app.main:app` → `POST /predict`.

---

## Results

> Numbers from `python train_model.py` on this repo (CPU). See `results/metrics.json`.

| Metric | Value |
|--------|------:|
| Train / test size | 18 / 6 |
| **Accuracy** | **0.8333** |
| Precision (spam) | 1.0000 |
| Recall (spam) | 0.6667 |
| F1 (spam) | 0.8000 |

**Honesty note:** This is a **tiny stub dataset** for an API demo — not Project 1's full SMS corpus. Scores will move if you swap in real data; the point is a reproducible train → serve → test loop with metrics written by a script.

---

## Project structure

```
model-api-fastapi/
├── README.md
├── HOW_I_BUILT_IT.md
├── BREAK_IT.md
├── requirements.txt
├── .gitignore
├── pytest.ini
├── train_model.py         # train stub → model/ + results/metrics.json
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml  # enable Actions: copy from ci/github-actions.yml
├── ci/github-actions.yml     # workflow YAML (on GitHub)
├── app/
│   ├── main.py            # FastAPI routes + latency middleware
│   ├── model_loader.py
│   └── logging_utils.py
├── tests/test_api.py
├── model/                 # spam_clf.joblib (gitignored)
└── results/metrics.json
```

---

## Setup & run

```bash
cd model-api-fastapi
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# train stub classifier → model/spam_clf.joblib + results/metrics.json
python train_model.py

# tests
pytest -q

# local API (http://127.0.0.1:8000/docs)
uvicorn app.main:app --reload --port 8000
```

Example:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"WIN free prize claim now http://bit.ly/x"}'
```

### Docker

```bash
docker compose up --build
# or: docker build -t model-api . && docker run -p 8000:8000 model-api
```

### CI note

Workflow YAML is published at `ci/github-actions.yml`. Copy it to `.github/workflows/ci.yml` (requires a GitHub token with the `workflow` scope to push that path) to enable Actions.

### Free-tier deploy (optional)

Works on **Railway**, **Render**, or **Fly.io** with the Dockerfile. Start command equivalent: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. This portfolio does not require a live deploy if you lack credentials — local uvicorn + CI is the bar.

---

## Learn by breaking it

- [HOW_I_BUILT_IT.md](HOW_I_BUILT_IT.md) — first-person build story
- [BREAK_IT.md](BREAK_IT.md) — intentional what-ifs, why it broke, how to fix

---

## License / notes

MIT. Educational portfolio project. Stub data only — not production spam filtering. Always regenerate the model with `train_model.py` before serving.

*Miles Robinson · Cybersecurity + AI Engineering · CPCC → UNC Charlotte · 2026*
