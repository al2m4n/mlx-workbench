"""
Audio Inference Service — wraps mlx-whisper for transcription.

mlx-whisper has a single-slot ModelHolder that the registry pre-warms via
`load_audio`. We pass the same `path_or_hf_repo` here so we hit that cache.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from services.metrics import InferenceMetric, MetricsBuffer, now_ts
from services.mlx_runtime import run_mlx
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioService:
    def __init__(self) -> None:
        self.metrics = MetricsBuffer.get()

    async def transcribe(
        self,
        model_key: str,
        model_id: str,
        audio_path: str,
        language: str | None = None,
        task: str = "transcribe",  # or "translate"
        word_timestamps: bool = False,
    ) -> dict[str, Any]:
        import mlx_whisper

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": model_id,
            "task": task,
            "word_timestamps": word_timestamps,
        }
        if language:
            kwargs["language"] = language

        t0 = time.perf_counter()
        try:
            result = await run_mlx(
                mlx_whisper.transcribe, audio_path, **kwargs
            )
        except TypeError as exc:
            # mlx-whisper defaults to fp16 mel features but several mlx-community
            # repos (e.g. whisper-tiny, whisper-base) ship float32 weights — the
            # encoder then returns float32 and decoding.py raises:
            #   "audio_features has an incorrect dtype: mlx.core.float32"
            # Retry with fp16=False so the dtype check expects float32.
            if "audio_features has an incorrect dtype" not in str(exc):
                raise
            logger.info(
                f"Retrying {model_id} in fp32 (model weights are not fp16)"
            )
            result = await run_mlx(
                mlx_whisper.transcribe, audio_path, fp16=False, **kwargs
            )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        text: str = result.get("text", "")
        segments = result.get("segments", []) or []
        detected_language = result.get("language")
        audio_duration_s = float(segments[-1]["end"]) if segments else 0.0

        await self.metrics.record(
            InferenceMetric(
                ts=now_ts(),
                modality="audio",
                model_key=model_key,
                latency_ms=latency_ms,
                generation_tokens=len(text.split()),
                extra={
                    "audio_duration_s": round(audio_duration_s, 2),
                    "language": detected_language,
                    "task": task,
                    "realtime_factor": (
                        round(audio_duration_s / (latency_ms / 1000), 2)
                        if latency_ms > 0
                        else None
                    ),
                },
            )
        )

        return {
            "text": text,
            "language": detected_language,
            "duration_s": round(audio_duration_s, 2),
            "latency_ms": latency_ms,
            "segments": [
                {
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "text": s["text"].strip(),
                }
                for s in segments
            ],
        }
