"""Plain JSON local storage for LLM API keys.

Keys are stored unencrypted in keys.json inside the app data directory.
Lookup order for any provider: keys.json → environment variable.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "ollama": ["OLLAMA_HOST"],
}

PROVIDERS = list(_PROVIDER_ENV.keys())


class KeyStore:
    def __init__(self, data_dir: Path):
        self._file = Path(data_dir) / "keys.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def get(self, provider: str) -> str | None:
        value = self._read().get(provider)
        if value:
            return value
        for env_var in _PROVIDER_ENV.get(provider, []):
            v = os.environ.get(env_var)
            if v:
                return v
        return None

    def set(self, provider: str, key: str):
        data = self._read()
        data[provider] = key
        self._write(data)

    def delete(self, provider: str):
        data = self._read()
        data.pop(provider, None)
        self._write(data)

    def status(self) -> dict:
        stored = self._read()
        result = {}
        for provider, env_vars in _PROVIDER_ENV.items():
            in_store = bool(stored.get(provider))
            in_env = any(os.environ.get(v) for v in env_vars)
            result[provider] = {
                "configured": in_store or in_env,
                "source": "store" if in_store else ("env" if in_env else None),
            }
        return result

    def _read(self) -> dict:
        if not self._file.exists():
            return {}
        try:
            return json.loads(self._file.read_text())
        except Exception as e:
            logger.warning("Could not read keys file: %s", e)
            return {}

    def _write(self, data: dict):
        self._file.write_text(json.dumps(data, indent=2))
