"""Audio Router — /audio/* (mlx-whisper transcription)."""

from __future__ import annotations

import json
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from services.audio_service import AudioService
from services.model_registry import ModelRegistry

router = APIRouter()
_service = AudioService()


class AudioLoadRequest(BaseModel):
    model_id: str = Field(..., examples=["mlx-community/whisper-tiny"])


@router.post("/load", summary="Pre-warm an mlx-whisper model")
async def load_audio(req: AudioLoadRequest):
    try:
        key = await ModelRegistry.get().load_audio(req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load: {exc}") from exc
    return {"status": "loaded", "model_key": key}


@router.post("/load-stream", summary="Load whisper model and stream progress as SSE")
async def load_audio_stream(req: AudioLoadRequest):
    registry = ModelRegistry.get()

    async def event_stream():
        try:
            async for ev in registry.load_audio_stream(req.model_id):
                yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


@router.post("/transcribe", summary="Transcribe an audio file")
async def transcribe(
    model_key: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    word_timestamps: bool = Form(False),
):
    if not ModelRegistry.get().is_loaded(model_key):
        raise HTTPException(404, f"Model not loaded: {model_key}")
    if task not in {"transcribe", "translate"}:
        raise HTTPException(400, "task must be 'transcribe' or 'translate'")

    meta = ModelRegistry.get().list_loaded()
    model_id = next(
        (m["model_id"] for m in meta if m["key"] == model_key),
        None,
    )
    if not model_id:
        raise HTTPException(404, f"Model metadata missing: {model_key}")

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()
        result = await _service.transcribe(
            model_key=model_key,
            model_id=model_id,
            audio_path=tmp.name,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
