"""In-process ring buffer for per-inference metrics."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable

_MAX_RECORDS = 1000


@dataclass(slots=True)
class InferenceMetric:
    ts: float
    modality: str
    model_key: str
    latency_ms: float
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_mb: float = 0.0
    finish_reason: str | None = None
    extra: dict | None = None


class MetricsBuffer:
    _instance: MetricsBuffer | None = None

    def __init__(self) -> None:
        self._records: deque[InferenceMetric] = deque(maxlen=_MAX_RECORDS)
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> MetricsBuffer:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def record(self, metric: InferenceMetric) -> None:
        async with self._lock:
            self._records.append(metric)

    def snapshot(self, model_key: str | None = None, limit: int = 100) -> list[dict]:
        # deque iteration is safe without the lock for a snapshot read.
        records: Iterable[InferenceMetric] = self._records
        if model_key:
            records = (r for r in records if r.model_key == model_key)
        # newest first, capped at `limit`
        out = list(records)[-limit:]
        out.reverse()
        return [asdict(r) for r in out]


def now_ts() -> float:
    return time.time()
