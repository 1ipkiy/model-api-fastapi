#!/usr/bin/env python3
"""Train a tiny TF-IDF + LogReg spam/ham stub and write metrics.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spam_clf.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"

# Tiny educational dataset (ham vs spam SMS-like). Enough to train a stub API model.
TEXTS = [
    ("hey are we still on for lunch", "ham"),
    ("can you pick up milk on the way home", "ham"),
    ("meeting moved to 3pm see you there", "ham"),
    ("thanks for the notes from class", "ham"),
    ("i will call you after work", "ham"),
    ("dont forget the dentist appointment tomorrow", "ham"),
    ("the project deadline is friday", "ham"),
    ("want to grab coffee later", "ham"),
    ("your package arrives today", "ham"),
    ("lets review the pull request tonight", "ham"),
    ("mom says dinner is at 6", "ham"),
    ("the report looks good to me", "ham"),
    ("WINNER claim your free iphone now click http://bit.ly/x", "spam"),
    ("URGENT your bank account will close verify http://secure-login.tk", "spam"),
    ("congratulations you won a $1000 gift card claim now", "spam"),
    ("free entry to win cash prize text WIN to 80085", "spam"),
    ("you have been selected for a free cruise call now", "spam"),
    ("act now limited offer click to claim your reward", "spam"),
    ("suspicious login detected verify password at http://evil.example", "spam"),
    ("get cheap meds online no prescription needed", "spam"),
    ("your warranty expires today renew at http://phish.example", "spam"),
    ("hot singles in your area click here", "spam"),
    ("you owe the IRS pay immediately or face arrest", "spam"),
    ("bitcoin double your money guaranteed invest now", "spam"),
]


def main() -> None:
    X = [t for t, _ in TEXTS]
    y = [label for _, label in TEXTS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "tfidf_logreg",
        "version": "1.0.0",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision_spam": round(float(precision_score(y_test, preds, pos_label="spam")), 4),
        "recall_spam": round(float(recall_score(y_test, preds, pos_label="spam")), 4),
        "f1_spam": round(float(f1_score(y_test, preds, pos_label="spam")), 4),
        "labels": ["ham", "spam"],
        "note": "Tiny stub dataset for API demo — not the full UCI SMS corpus from Project 1.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "labels": ["ham", "spam"],
        "model_name": metrics["model_name"],
        "version": metrics["version"],
    }
    joblib.dump(bundle, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(json.dumps({k: metrics[k] for k in ("accuracy", "precision_spam", "recall_spam", "f1_spam", "n_train", "n_test")}, indent=2))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
