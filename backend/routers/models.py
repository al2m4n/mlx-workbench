"""Models Router — /models/*"""

from fastapi import APIRouter, HTTPException

from services.model_registry import ModelRegistry

router = APIRouter()


@router.get("/", summary="List all loaded models")
async def list_all():
    return ModelRegistry.get().list_loaded()


@router.delete("/{model_key:path}", summary="Unload a model")
async def unload(model_key: str):
    ok = await ModelRegistry.get().unload(model_key)
    if not ok:
        raise HTTPException(404, f"Model not found: {model_key}")
    return {"status": "unloaded", "model_key": model_key}
