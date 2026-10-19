"""FastAPI app: tiny spam/ham text classifier API."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.logging_utils import Timer, log_request
from app.model_loader import load_model, model_exists, predict_proba_spam

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if model_exists():
        load_model()
    yield


app = FastAPI(
    title="Model API (Spam/Ham)",
    description="CPU-friendly FastAPI serving a tiny TF-IDF + Logistic Regression classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["WIN free prize now!!!"])


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    model: str
    version: str
    latency_ms: float


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Latency-Ms"] = f"{ms:.2f}"
    if request.url.path.startswith("/predict"):
        log_request(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": round(ms, 3),
                # toy "cost" proxy: 0.0001 units per ms (docs/demo only)
                "cost_units": round(ms * 0.0001, 6),
            }
        )
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model_exists(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Model API — see /docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if not model_exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python train_model.py")
    with Timer() as t:
        try:
            out = predict_proba_spam(body.text)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e
    return PredictResponse(latency_ms=round(t.ms, 3), **out)
