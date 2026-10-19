# Break It: Intentional Failure Lab (Model API)

**Author:** Miles Robinson · **2026**

Break serving and training on purpose, then restore and re-test.

**Rules:**

1. Branch or stash before big edits.
2. After each experiment: `python train_model.py && pytest -q`.
3. Do not invent metrics — read `results/metrics.json`.

---

## Experiment 1 — Delete the model artifact

### What I change
```bash
rm model/spam_clf.joblib
uvicorn app.main:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"hi"}'
```

### What I expect vs what happens
- **Expect:** Soft failure.
- **Happens:** HTTP 503 with a message to run `train_model.py`.

### Why it broke
Serving assumes an artifact exists.

### How to fix it
`python train_model.py`

### What I learned
Health vs ready: `/health` can say `model_loaded: false`; predict must fail loudly.

---

## Experiment 2 — Remove `min_length=1` validation

### What I change
Allow empty `text` in `PredictRequest`.

### What I expect vs what happens
- **Expect:** Harmless.
- **Happens:** Empty strings hit the vectorizer; behavior is undefined for users.

### Why it broke
API contracts should reject garbage early.

### How to fix it
Restore `min_length=1` (422 on empty).

### What I learned
Validation is part of model quality.

---

## Experiment 3 — Flip train/test labels randomly

### What I change
Shuffle `y` before fit in `train_model.py`.

### What I expect vs what happens
- **Expect:** Same API shape.
- **Happens:** Metrics collapse; predictions look random.

### Why it broke
Serving a broken model still "works" HTTP-wise.

### How to fix it
Restore labels; gate deploys on metric floors in CI.

### What I learned
API green ≠ model green. Keep `metrics.json` in the loop.

---

## Experiment 4 — Disable latency middleware

### What I change
Comment out the timing middleware.

### What I expect vs what happens
- **Expect:** Faster code path.
- **Happens:** No `X-Latency-Ms`, no JSONL cost trail — you lose the operational story.

### Why it broke
Observability isn't optional in production narratives.

### How to fix it
Restore middleware + `log_request`.

### What I learned
Portfolio APIs should demonstrate how you'd debug latency.

---

## Experiment 5 — Break CI by skipping train

### What I change
Remove the "Train stub model" step from `.github/workflows/ci.yml` and delete `model/`.

### What I expect vs what happens
- **Expect:** Tests still pass somehow.
- **Happens:** Fixture trains locally, but forgetting train in a lean CI job fails if the fixture changes — better to keep an explicit train step.

### Why it broke
Hidden setup diverges from what CI documents.

### How to fix it
Keep explicit `python train_model.py` before `pytest` in CI.

### What I learned
Make the happy path obvious in YAML.

---

## Restore checklist

```bash
git checkout -- train_model.py app/main.py .github/workflows/ci.yml
python train_model.py
pytest -q
```

*Miles Robinson · 2026*
