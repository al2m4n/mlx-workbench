"""
Hooks huggingface_hub's tqdm into an asyncio queue so model downloads can
stream progress events to SSE clients instead of just printing bars to stderr.

Also implements cooperative cancellation: every tqdm subclass instance checks
a shared `threading.Event` before emitting, and raises `LoadCancelled` if the
event is set. This breaks `snapshot_download`'s blocking loop at the next
chunk boundary, which is the only way to stop a hf_hub download in flight
(the underlying C calls aren't interruptible).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from tqdm import tqdm


class LoadCancelled(Exception):
    """Raised inside the MLX worker thread when the SSE client disconnects
    or the user clicks Cancel. Propagates up out of `snapshot_download` so
    the worker thread exits and frees the single-slot MLX executor."""


def _clean_desc(desc: Any) -> str:
    """Map hf_hub's tqdm descriptions to something a user wants to read."""
    s = str(desc or "").strip()
    if not s:
        return "downloading"
    # hf_hub uses this generic string when the total is being negotiated;
    # the per-file context is lost, so flatten it.
    if "incomplete total" in s.lower():
        return "downloading file"
    return s


def make_progress_tqdm(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
    cancel_event: threading.Event | None = None,
) -> type[tqdm]:
    """Build a tqdm subclass bound to a specific event loop and queue.

    `huggingface_hub.snapshot_download(..., tqdm_class=...)` instantiates one
    bar per file. Each instance throttles its own emissions: a `progress`
    event fires only if the percent advanced ≥1.0 or ≥250 ms passed since
    the last emit. A final emit is forced on close so 100 % always lands.

    If `cancel_event` is set, the next `update()` raises `LoadCancelled` so
    the surrounding `snapshot_download` loop exits.
    """

    MIN_PERCENT_DELTA = 1.0
    MIN_INTERVAL_S = 0.25

    class ProgressTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._last_percent: float = -1.0
            self._last_emit_ts: float = 0.0

        def _emit(self, force: bool = False) -> None:
            total = self.total
            if not total:
                return
            # hf_hub occasionally bumps `n` past `total` mid-stream when its
            # estimate of the file size is revised; clamp so the UI never sees
            # >100 %.
            percent = round(min(100.0, 100.0 * self.n / total), 1)
            now = time.monotonic()
            if not force:
                if (
                    percent - self._last_percent < MIN_PERCENT_DELTA
                    and now - self._last_emit_ts < MIN_INTERVAL_S
                ):
                    return
            self._last_percent = percent
            self._last_emit_ts = now
            payload = {
                "event": "progress",
                "data": {
                    "file": _clean_desc(self.desc),
                    "downloaded": int(self.n),
                    "total": int(total),
                    "percent": percent,
                },
            }
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except RuntimeError:
                pass

        def update(self, n: int = 1) -> bool | None:
            if cancel_event is not None and cancel_event.is_set():
                raise LoadCancelled()
            displayed = super().update(n)
            self._emit()
            return displayed

        def close(self) -> None:
            try:
                self._emit(force=True)
            finally:
                super().close()

    return ProgressTqdm
