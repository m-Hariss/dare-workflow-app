"""Encrypted local storage for LLM API keys.

Stores provider credentials as a Fernet-encrypted JSON blob in the app's
data directory. A machine-specific master key is generated on first run and
written alongside the secrets file with chmod 600.

Lookup order for any provider: KeyStore file → environment variable.
"""
import json
import logging
import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Maps the canonical provider name used here to the env var(s) the LLMClient
# already knows about, so the store falls back gracefully when no UI key exists.
_PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "ollama": ["OLLAMA_HOST"],
}

PROVIDERS = list(_PROVIDER_ENV.keys())


class KeyStore:
    def __init__(self, data_dir: Path):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._master_key_file = self._dir / ".master.key"
        self._secrets_file = self._dir / "secrets.json"
        self._fernet = Fernet(self._load_or_create_master_key())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, provider: str) -> str | None:
        """Return the stored key for provider, or fall back to env var."""
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
        """Return {provider: {"configured": bool, "source": "store"|"env"|None}}."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_create_master_key(self) -> bytes:
        if self._master_key_file.exists():
            return self._master_key_file.read_bytes().strip()
        key = Fernet.generate_key()
        self._master_key_file.write_bytes(key)
        os.chmod(self._master_key_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
        logger.info("Generated new master key at %s", self._master_key_file)
        return key

    def _read(self) -> dict:
        if not self._secrets_file.exists():
            return {}
        try:
            return json.loads(self._fernet.decrypt(self._secrets_file.read_bytes()))
        except (InvalidToken, Exception) as e:
            logger.warning("Could not decrypt secrets file: %s", e)
            return {}

    def _write(self, data: dict):
        encrypted = self._fernet.encrypt(json.dumps(data).encode())
        self._secrets_file.write_bytes(encrypted)
        os.chmod(self._secrets_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
