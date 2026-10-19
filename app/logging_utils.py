"""Request latency / cost-style logging helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "request_log.jsonl"


def log_request(event: dict[str, Any], path: Path | None = None) -> None:
    target = path or LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        f.write(json.dumps(event) + "\n")


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.ms = (time.perf_counter() - self.start) * 1000.0
