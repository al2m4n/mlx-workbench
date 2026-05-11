"""
LLM Inference Service — wraps mlx-lm with chat templating, streaming,
and metrics capture.

Generation runs on the MLX (Metal) engine which is single-threaded per process.
We push the call onto a worker thread with `asyncio.to_thread` so the event
loop stays responsive for SSE flushing and concurrent requests.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from services.metrics import InferenceMetric, MetricsBuffer, now_ts
from services.mlx_runtime import run_mlx
from services.model_registry import ModelRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatMessage(dict):
    role: str
    content: str


def _format_chat(tokenizer, messages: list[dict]) -> str:
    """Apply the model's chat template, falling back to a plain join."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        logger.warning(f"Chat template failed ({exc}); falling back to join")
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class LLMService:
    def __init__(self) -> None:
        self.registry = ModelRegistry.get()
        self.metrics = MetricsBuffer.get()

    async def chat(
        self,
        model_key: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Non-streaming chat — returns the full response in one shot."""
        chunks: list[str] = []
        meta: dict = {}
        async for event in self.stream(
            model_key, messages, max_tokens, temperature, top_p
        ):
            if event["event"] == "token":
                chunks.append(event["data"]["text"])
            elif event["event"] == "done":
                meta = event["data"]
        return {"text": "".join(chunks), **meta}

    async def stream(
        self,
        model_key: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[dict, None]:
        """Yield SSE-shaped events: `token` per delta, then a final `done`.

        Bridges the synchronous `stream_generate` generator (which runs on a
        worker thread) into an async stream via an asyncio.Queue so the FastAPI
        event loop can flush each chunk to the client.
        """
        model = self.registry.get_model(model_key)
        tokenizer = self.registry.get_tokenizer(model_key)
        if model is None or tokenizer is None:
            raise LookupError(f"Model not loaded: {model_key}")

        prompt = _format_chat(tokenizer, messages)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _produce() -> None:
            try:
                for resp in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, resp)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        t0 = time.perf_counter()
        worker = asyncio.create_task(run_mlx(_produce))
        last = None

        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                last = item
                if item.text:
                    yield {"event": "token", "data": {"text": item.text}}
        finally:
            await worker

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if last is None:
            done_payload = {
                "model_key": model_key,
                "latency_ms": latency_ms,
                "finish_reason": "no_output",
            }
        else:
            done_payload = {
                "model_key": model_key,
                "latency_ms": latency_ms,
                "prompt_tokens": last.prompt_tokens,
                "generation_tokens": last.generation_tokens,
                "prompt_tps": round(last.prompt_tps, 2),
                "generation_tps": round(last.generation_tps, 2),
                "peak_memory_mb": round(last.peak_memory, 2),
                "finish_reason": last.finish_reason,
            }
            await self.metrics.record(
                InferenceMetric(
                    ts=now_ts(),
                    modality="llm",
                    model_key=model_key,
                    prompt_tokens=last.prompt_tokens,
                    generation_tokens=last.generation_tokens,
                    prompt_tps=round(last.prompt_tps, 2),
                    generation_tps=round(last.generation_tps, 2),
                    latency_ms=latency_ms,
                    peak_memory_mb=round(last.peak_memory, 2),
                    finish_reason=last.finish_reason,
                )
            )

        yield {"event": "done", "data": done_payload}
