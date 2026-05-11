"""Health Router — /health/*"""

import platform
import sys
import time

from fastapi import APIRouter

router = APIRouter()
_start = time.time()


@router.get("/", summary="Liveness probe")
async def health():
    return {"status": "ok", "uptime_s": round(time.time() - _start, 1)}


@router.get("/ready", summary="Readiness probe")
async def ready():
    return {"status": "ready"}


@router.get("/info", summary="System and runtime info")
async def info():
    mlx_version = "unavailable"
    try:
        import mlx.core as mx  # type: ignore
        mlx_version = getattr(mx, "__version__", "unknown")
    except ImportError:
        pass

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "Apple Silicon",
        "mlx_version": mlx_version,
    }
