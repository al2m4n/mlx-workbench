"""LLM Router — /llm/*"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from services.llm_service import LLMService
from services.metrics import MetricsBuffer
from services.model_registry import ModelRegistry

router = APIRouter()
_service = LLMService()


class LoadRequest(BaseModel):
    model_id: str = Field(..., examples=["mlx-community/Llama-3.2-3B-Instruct-4bit"])


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_key: str
    messages: list[ChatMessageIn]
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = True


@router.post("/load", summary="Load an MLX-format LLM from HuggingFace")
async def load_model(req: LoadRequest):
    registry = ModelRegistry.get()
    try:
        key = await registry.load_llm(req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load: {exc}") from exc
    return {"status": "loaded", "model_key": key}


@router.post("/load-stream", summary="Load an LLM and stream progress as SSE")
async def load_model_stream(req: LoadRequest):
    registry = ModelRegistry.get()

    async def event_stream():
        try:
            async for ev in registry.load_llm_stream(req.model_id):
                yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


@router.post("/chat", summary="Chat with a loaded LLM (streaming or blocking)")
async def chat(req: ChatRequest):
    if not ModelRegistry.get().is_loaded(req.model_key):
        raise HTTPException(404, f"Model not loaded: {req.model_key}")

    messages = [m.model_dump() for m in req.messages]

    if not req.stream:
        result = await _service.chat(
            req.model_key,
            messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return result

    async def event_stream():
        try:
            async for event in _service.stream(
                req.model_key,
                messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                yield {"event": event["event"], "data": json.dumps(event["data"])}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_stream())


@router.get("/metrics", summary="Recent inference metrics (newest first)")
async def metrics(
    model_key: str | None = Query(None, description="Filter by model key"),
    limit: int = Query(100, ge=1, le=1000),
):
    return MetricsBuffer.get().snapshot(model_key=model_key, limit=limit)


@router.get("/loaded", summary="List loaded LLM models")
async def loaded():
    return [
        m for m in ModelRegistry.get().list_loaded() if m.get("modality") == "llm"
    ]
