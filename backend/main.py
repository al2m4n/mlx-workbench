"""MLX Inference Server — LLM · Audio · Vision on Apple Silicon."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import llm, audio, video, models, health
from services.mlx_runtime import prewarm as mlx_prewarm
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLX Inference Server")
    await mlx_prewarm()
    logger.info("MLX runtime ready.")
    yield
    logger.info("Shutting down MLX Inference Server")


app = FastAPI(
    title="MLX Inference Server",
    description="Apple Silicon inference interface for HuggingFace models (LLM · Audio · Video)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router,  prefix="/health",  tags=["Health"])
app.include_router(models.router,  prefix="/models",  tags=["Models"])
app.include_router(llm.router,     prefix="/llm",     tags=["LLM"])
app.include_router(audio.router,   prefix="/audio",   tags=["Audio"])
app.include_router(video.router,   prefix="/video",   tags=["Video"])


@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Duration-Ms"] = str(duration)
    logger.info(f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
