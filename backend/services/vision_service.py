"""
Vision Inference Service — wraps mlx-vlm for image and video understanding.

The same VLM accepts either an image or a video file via the `image` kwarg
(despite the name — mlx-vlm uses opencv internally to extract frames from
video paths). We expose a single `generate()` here; streaming for VLMs is
deferred since `mlx_vlm.stream_generate` has different chunking behaviour
across model families and the blocking call is fine for description-style
prompts.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from services.metrics import InferenceMetric, MetricsBuffer, now_ts
from services.mlx_runtime import run_mlx
from services.model_registry import ModelRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


class VisionService:
    def __init__(self) -> None:
        self.registry = ModelRegistry.get()
        self.metrics = MetricsBuffer.get()

    async def generate(
        self,
        model_key: str,
        prompt: str,
        media_path: str,
        media_kind: str,  # "image" or "video"
        max_tokens: int = 512,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        model = self.registry.get_model(model_key)
        processor = self.registry.get_tokenizer(model_key)
        if model is None or processor is None:
            raise LookupError(f"Model not loaded: {model_key}")

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            processor, model.config, prompt, num_images=1
        )

        t0 = time.perf_counter()
        result = await run_mlx(
            generate,
            model,
            processor,
            formatted,
            image=[media_path],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # mlx_vlm.generate returns a GenerationResult dataclass.
        text = getattr(result, "text", "") or ""
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        prompt_tps = float(getattr(result, "prompt_tps", 0.0) or 0.0)
        generation_tps = float(getattr(result, "generation_tps", 0.0) or 0.0)
        peak_memory = float(getattr(result, "peak_memory", 0.0) or 0.0)

        await self.metrics.record(
            InferenceMetric(
                ts=now_ts(),
                modality="video",
                model_key=model_key,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                prompt_tps=round(prompt_tps, 2),
                generation_tps=round(generation_tps, 2),
                peak_memory_mb=round(peak_memory, 2),
                extra={"media_kind": media_kind},
            )
        )

        return {
            "text": text,
            "model_key": model_key,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "prompt_tps": round(prompt_tps, 2),
            "generation_tps": round(generation_tps, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "media_kind": media_kind,
        }
