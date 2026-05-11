"""Video Router — /video/* (mlx-vlm vision-language models, image or video)."""

from __future__ import annotations

import json
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from services.model_registry import ModelRegistry
from services.vision_service import VisionService

router = APIRouter()
_service = VisionService()

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


class VisionLoadRequest(BaseModel):
    model_id: str = Field(..., examples=["mlx-community/SmolVLM-256M-Instruct-bf16"])


@router.post("/load", summary="Load an mlx-vlm vision-language model")
async def load_video(req: VisionLoadRequest):
    try:
        key = await ModelRegistry.get().load_video(req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load: {exc}") from exc
    return {"status": "loaded", "model_key": key}


@router.post("/load-stream", summary="Load vision model and stream progress as SSE")
async def load_video_stream(req: VisionLoadRequest):
    registry = ModelRegistry.get()

    async def event_stream():
        try:
            async for ev in registry.load_video_stream(req.model_id):
                yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


@router.post(
    "/generate",
    summary="Generate a response about an image or video",
)
async def generate(
    model_key: str = Form(...),
    prompt: str = Form("Describe this in detail."),
    file: UploadFile = File(...),
    max_tokens: int = Form(512),
    temperature: float = Form(0.4),
):
    if not ModelRegistry.get().is_loaded(model_key):
        raise HTTPException(404, f"Model not loaded: {model_key}")

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix in _IMAGE_EXTS:
        media_kind = "image"
    elif suffix in _VIDEO_EXTS:
        media_kind = "video"
    else:
        raise HTTPException(
            400,
            f"Unsupported file type {suffix!r}; expected image or video.",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.close()
        result = await _service.generate(
            model_key=model_key,
            prompt=prompt,
            media_path=tmp.name,
            media_kind=media_kind,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
