"""Bundles the external service clients handlers depend on."""
import os
from typing import Optional

from clients.dare_client import DareClient
from clients.llm_client import LLMClient


class Services:
    def __init__(self, dare: DareClient, llm: LLMClient):
        self.dare = dare
        self.llm = llm

    @classmethod
    def from_config(cls, base_url: str | None = None, token: str | None = None, key_store=None):
        base_url = base_url or os.environ.get("DARE_BASE_URL", "http://localhost:8000")
        token = token or os.environ.get("DARE_TOKEN", "")

        api_keys: dict = {}
        ollama_host: Optional[str] = None

        if key_store is not None:
            for provider in ("openai", "claude", "gemini"):
                v = key_store.get(provider)
                if v:
                    api_keys[provider] = v
            ollama_host = key_store.get("ollama")

        return cls(
            dare=DareClient(base_url, token),
            llm=LLMClient(api_keys=api_keys or None, ollama_host=ollama_host),
        )
