"""Bundles the external services handlers depend on (Dare + LLM)."""
import os

from clients.dare_client import DareClient
from clients.llm_client import LLMClient


class Services:
    def __init__(self, dare: DareClient, llm: LLMClient):
        self.dare = dare
        self.llm = llm

    @classmethod
    def from_config(cls, base_url: str | None = None, token: str | None = None):
        base_url = base_url or os.environ.get("DARE_BASE_URL", "http://localhost:8000")
        token = token or os.environ.get("DARE_TOKEN", "")
        return cls(dare=DareClient(base_url, token), llm=LLMClient())
