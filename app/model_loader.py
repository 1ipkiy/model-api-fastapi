"""Load the trained sklearn pipeline from model/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = MODEL_DIR / "spam_clf.joblib"

_bundle: dict[str, Any] | None = None


def model_exists() -> bool:
    return MODEL_PATH.exists()


def load_model(force: bool = False) -> dict[str, Any]:
    global _bundle
    if _bundle is not None and not force:
        return _bundle
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run: python train_model.py"
        )
    _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_proba_spam(text: str) -> dict[str, Any]:
    bundle = load_model()
    pipe = bundle["pipeline"]
    labels = bundle["labels"]  # ["ham", "spam"]
    proba = pipe.predict_proba([text])[0]
    pred_idx = int(proba.argmax())
    return {
        "label": labels[pred_idx],
        "probabilities": {labels[i]: float(proba[i]) for i in range(len(labels))},
        "model": bundle.get("model_name", "tfidf_logreg"),
        "version": bundle.get("version", "1.0.0"),
    }
