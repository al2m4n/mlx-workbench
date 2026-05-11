"""
Model Registry — singleton holding loaded MLX models for all modalities.

Models are keyed by `<modality>::<model_id>` in a shared dict.

Notes
-----
- mlx-whisper has its own single-slot `ModelHolder` cache; loading a second
  whisper model evicts the first. We pre-warm it on `load_audio` and treat
  the registry entry as metadata-only.
- mlx-vlm has no internal cache, so we hold the (model, processor) tuple
  ourselves and support multiple loaded VLMs simultaneously.
"""

from __future__ import annotations

import asyncio
import threading
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from services.load_progress import LoadCancelled, make_progress_tqdm
from services.mlx_runtime import run_mlx
from utils.logger import get_logger

logger = get_logger(__name__)


class Modality(str, Enum):
    LLM = "llm"
    AUDIO = "audio"
    VIDEO = "video"


class ModelRegistry:
    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._metadata: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def load_llm(self, model_id: str) -> str:
        key = f"llm::{model_id}"
        async with self._lock:
            if key in self._models:
                logger.info(f"Cache hit: {key}")
                return key

            logger.info(f"Loading LLM: {model_id}")
            from mlx_lm import load

            model, tokenizer = await run_mlx(load, model_id)
            self._models[key] = model
            self._tokenizers[key] = tokenizer
            self._metadata[key] = {
                "model_id": model_id,
                "modality": Modality.LLM.value,
            }
            logger.info(f"Loaded: {key}")
        return key

    async def load_audio(self, model_id: str) -> str:
        """Pre-warm an mlx-whisper model via its ModelHolder cache.

        Only one whisper model is held at a time by the underlying library —
        loading a second one evicts the first.
        """
        key = f"audio::{model_id}"
        async with self._lock:
            if key in self._models:
                logger.info(f"Cache hit: {key}")
                return key

            logger.info(f"Loading audio model: {model_id}")
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            await run_mlx(ModelHolder.get_model, model_id, mx.float32)
            # Sentinel: mlx-whisper owns the actual reference, but we mark it
            # loaded so `is_loaded` and `list_loaded` work uniformly.
            self._models[key] = True
            self._metadata[key] = {
                "model_id": model_id,
                "modality": Modality.AUDIO.value,
            }
            logger.info(f"Loaded: {key}")
        return key

    async def load_video(self, model_id: str) -> str:
        """Load an mlx-vlm vision-language model (used for both image and video)."""
        key = f"video::{model_id}"
        async with self._lock:
            if key in self._models:
                logger.info(f"Cache hit: {key}")
                return key

            logger.info(f"Loading vision model: {model_id}")
            from mlx_vlm import load as vlm_load

            model, processor = await run_mlx(vlm_load, model_id)
            self._models[key] = model
            self._tokenizers[key] = processor
            self._metadata[key] = {
                "model_id": model_id,
                "modality": Modality.VIDEO.value,
            }
            logger.info(f"Loaded: {key}")
        return key

    async def _load_stream(
        self,
        key: str,
        model_id: str,
        modality: Modality,
        loader: Callable[[], tuple[Any, Any]],
    ) -> AsyncGenerator[dict, None]:
        """Shared driver for the modality-specific streaming loaders.

        `loader` is a sync callable that runs on the MLX worker thread *after*
        the HuggingFace download finishes. It must return `(model, tokenizer)`
        — modalities without a separate tokenizer (audio) return `(sentinel, None)`.
        """
        async with self._lock:
            if key in self._models:
                logger.info(f"Cache hit: {key}")
                yield {
                    "event": "done",
                    "data": {"model_key": key, "cached": True},
                }
                return

            logger.info(f"Loading {modality.value}: {model_id}")
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()
            cancel_event = threading.Event()
            ProgressTqdm = make_progress_tqdm(loop, queue, cancel_event)

            def _download_then_load() -> None:
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(model_id, tqdm_class=ProgressTqdm)
                    if cancel_event.is_set():
                        raise LoadCancelled()
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "phase", "data": {"phase": "loading_weights"}},
                    )
                    model, tokenizer = loader()
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("__result__", model, tokenizer),
                    )
                except LoadCancelled:
                    logger.info(f"Load cancelled: {key}")
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            yield {"event": "phase", "data": {"phase": "downloading"}}
            worker = asyncio.create_task(run_mlx(_download_then_load))
            result: tuple | None = None
            try:
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if isinstance(item, tuple) and item and item[0] == "__result__":
                        result = item
                        continue
                    yield item
            finally:
                # On any exit (normal, error, or client-disconnect cancel),
                # signal the worker to stop at the next chunk. We deliberately
                # do NOT await `worker` here: awaiting from a cancelled task
                # re-raises CancelledError, and the worker will free the MLX
                # executor on its own a moment later — the next load queues
                # behind it on the single-slot pool.
                cancel_event.set()
                if not worker.done():
                    worker.add_done_callback(lambda t: t.exception())

            if result is None:
                # Cancelled or otherwise interrupted before the loader returned.
                return

            _, model, tokenizer = result
            self._models[key] = model
            if tokenizer is not None:
                self._tokenizers[key] = tokenizer
            self._metadata[key] = {
                "model_id": model_id,
                "modality": modality.value,
            }
            logger.info(f"Loaded: {key}")

        yield {"event": "done", "data": {"model_key": key}}

    def load_llm_stream(self, model_id: str) -> AsyncGenerator[dict, None]:
        def _loader() -> tuple[Any, Any]:
            from mlx_lm import load

            return load(model_id)

        return self._load_stream(
            f"llm::{model_id}", model_id, Modality.LLM, _loader
        )

    def load_audio_stream(self, model_id: str) -> AsyncGenerator[dict, None]:
        def _loader() -> tuple[Any, Any]:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.get_model(model_id, mx.float32)
            # Sentinel matches `load_audio` above — mlx-whisper owns the ref.
            return True, None

        return self._load_stream(
            f"audio::{model_id}", model_id, Modality.AUDIO, _loader
        )

    def load_video_stream(self, model_id: str) -> AsyncGenerator[dict, None]:
        def _loader() -> tuple[Any, Any]:
            from mlx_vlm import load as vlm_load

            return vlm_load(model_id)

        return self._load_stream(
            f"video::{model_id}", model_id, Modality.VIDEO, _loader
        )

    def get_model(self, key: str) -> Any:
        return self._models.get(key)

    def get_tokenizer(self, key: str) -> Any:
        return self._tokenizers.get(key)

    def list_loaded(self) -> list[dict]:
        return [{"key": k, **v} for k, v in self._metadata.items()]

    def is_loaded(self, key: str) -> bool:
        return key in self._models

    async def unload(self, key: str) -> bool:
        async with self._lock:
            if key not in self._models:
                return False
            del self._models[key]
            self._tokenizers.pop(key, None)
            del self._metadata[key]
            logger.info(f"Unloaded: {key}")
            return True
