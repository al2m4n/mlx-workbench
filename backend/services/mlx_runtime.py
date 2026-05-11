"""
Single-threaded executor for all MLX inference work.

Why this exists:
    mlx >= 0.31 makes GPU streams thread-local. asyncio.to_thread() uses the
    default thread pool where each call may land on a different worker — the
    first call initialises a Stream(gpu, 0), the second hits a fresh thread
    and crashes with `std::runtime_error: There is no Stream(gpu, 0) in
    current thread`.

    Pin every MLX call to one dedicated worker so the stream is created once
    and reused. This also serialises MLX work, which is fine because MLX is
    single-process / single-stream anyway.

Use it like:
    from services.mlx_runtime import run_mlx
    result = await run_mlx(some_blocking_mlx_fn, arg1, arg2, kw=...)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _init_thread() -> None:
    """Pin every MLX-touching library to this worker thread.

    Two things have to happen on the same thread:
      1. The default GPU stream (`Stream(gpu, 0)`) must be registered.
      2. Any module that creates a stream at import time (`mlx_vlm` does,
         see mlx_vlm/generate.py: `generation_stream = mx.new_stream(...)`)
         must be imported here, otherwise the stream is bound to whichever
         thread happened to import it first.

    Because Python caches modules in sys.modules, importing them here before
    any other thread does the import means every subsequent
    `from mlx_vlm import ...` is a no-op that returns the cached module —
    streams stay on this thread. `prewarm()` below runs this initializer at
    application startup to win the import race.
    """
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        # Touch the GPU so Stream(gpu, 0) is allocated on this thread.
        mx.eval(mx.array([0]))
    except Exception:
        # MLX not available (e.g. on Linux for tests) — fall back silently.
        pass

    for mod in ("mlx_lm", "mlx_whisper", "mlx_vlm"):
        try:
            __import__(mod)
        except Exception:
            # Optional deps; if a library isn't installed, skip it.
            pass


_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="mlx",
    initializer=_init_thread,
)


async def run_mlx(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, partial(fn, *args, **kwargs))


async def prewarm() -> None:
    """Force the MLX worker thread to spin up *now* so its initializer runs
    before anything else can import mlx_vlm/mlx_lm/mlx_whisper on the wrong
    thread. Call once at FastAPI lifespan startup."""
    await run_mlx(lambda: None)
