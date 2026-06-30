"""LLM key management endpoints."""
from fastapi import HTTPException
from pydantic import BaseModel

from app_state import key_store


class SaveKeyRequest(BaseModel):
    provider: str
    key: str


def register(app):
    @app.get("/keys/status", tags=["syftbox"])
    def keys_status():
        return key_store.status()

    @app.post("/keys", tags=["syftbox"])
    def save_key(body: SaveKeyRequest):
        if not body.provider or not body.key:
            raise HTTPException(status_code=400, detail="provider and key are required")
        key_store.set(body.provider.lower(), body.key.strip())
        return {"ok": True, "provider": body.provider.lower()}

    @app.delete("/keys/{provider}", tags=["syftbox"])
    def delete_key(provider: str):
        key_store.delete(provider.lower())
        return {"ok": True, "provider": provider.lower()}
