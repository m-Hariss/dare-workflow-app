"""Multi-provider LLM client (no SDKs — plain HTTP via requests).

Mirrors Dare's provider behavior:
- openai / custom / LiteLLM proxy : OpenAI-compatible /chat/completions
                                    (+ /responses for web search)
- claude (Anthropic)             : /v1/messages
- gemini (Google)                : /v1beta/models/{model}:generateContent
- llama (Ollama, local)          : /api/chat

Credentials come from the runner's environment, never from the exported workflow.
All calls are non-streaming (the runner collects the full text).
"""
import os

import requests

OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TOKEN_BUFFER = 3000  # Dare adds this to work around a Gemini SDK bug

# Provider -> env var names to try (first hit wins).
ENV_KEYS_BY_PROVIDER = {
    "openai": ["OPENAI_API_KEY"],
    "custom": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
}


class LLMClient:
    def __init__(self, api_keys: dict | None = None, timeout: int = 180, ollama_host: str | None = None):
        self.api_keys = api_keys or {}
        self.timeout = timeout
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def complete(self, llm: dict, message: str, max_tokens=None, temperature=None, web_search=False):
        """Return (text, usage_dict). Dispatches to the right provider."""
        provider = (llm.get("provider") or "openai").lower()
        base_url = llm.get("baseUrl")

        # A custom base_url means an OpenAI-compatible proxy (e.g. LiteLLM) —
        # route everything through the OpenAI path regardless of provider.
        if base_url:
            return self._openai(llm, message, max_tokens, temperature, web_search, base_url=base_url)
        if provider in ("openai", "custom"):
            return self._openai(llm, message, max_tokens, temperature, web_search)
        if provider in ("claude", "anthropic"):
            return self._claude(llm, message, max_tokens, temperature, web_search)
        if provider in ("gemini", "google"):
            return self._gemini(llm, message, max_tokens, temperature, web_search)
        if provider in ("llama", "ollama"):
            return self._llama(llm, message, max_tokens, temperature)
        # Unknown provider: best-effort OpenAI-compatible.
        return self._openai(llm, message, max_tokens, temperature, web_search)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _api_key(self, provider: str):
        if provider in self.api_keys:
            return self.api_keys[provider]
        for env_name in ENV_KEYS_BY_PROVIDER.get(provider, []):
            if os.environ.get(env_name):
                return os.environ[env_name]
        return None

    def _temp(self, llm, temperature):
        if temperature is not None and llm.get("supportsTemperature", True):
            return temperature
        return None

    @staticmethod
    def _error_detail(resp) -> str:
        detail = resp.text
        try:
            body = resp.json()
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = err.get("message") or detail
        except ValueError:
            pass
        return detail

    def _post(self, url, *, json=None, headers=None):
        return requests.post(url, json=json, headers=headers, timeout=self.timeout)

    def _raise(self, resp, model, where):
        raise RuntimeError(
            f"LLM call failed ({resp.status_code}) for model '{model}' at {where}: "
            f"{self._error_detail(resp)}"
        )

    # ------------------------------------------------------------------
    # OpenAI / custom / LiteLLM (OpenAI-compatible)
    # ------------------------------------------------------------------

    def _openai(self, llm, message, max_tokens, temperature, web_search, base_url=None):
        provider = (llm.get("provider") or "openai").lower()
        model = llm.get("identifier")
        base = (base_url or OPENAI_DEFAULT_BASE).rstrip("/")
        key = self._api_key("openai") if provider in ("openai", "custom") else (self._api_key(provider) or self._api_key("openai"))
        if not key:
            raise RuntimeError(f"No API key available for provider '{provider}'")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        # Web search uses the Responses API.
        if web_search:
            body = {"model": model, "input": message, "tools": [{"type": "web_search"}]}
            if max_tokens:
                body["max_output_tokens"] = max_tokens
            resp = self._post(f"{base}/responses", json=body, headers=headers)
            if not resp.ok:
                self._raise(resp, model, f"{base}/responses")
            return self._extract_responses_api(resp.json())

        base_body = {"model": model, "messages": [{"role": "user", "content": message}]}
        temp = self._temp(llm, temperature)
        if temp is not None:
            base_body["temperature"] = temp

        def _attempt(token_param):
            body = dict(base_body)
            if max_tokens:
                body[token_param] = max_tokens
            return self._post(f"{base}/chat/completions", json=body, headers=headers)

        # Newer models require max_completion_tokens; older accept max_tokens.
        resp = _attempt("max_completion_tokens")
        if not resp.ok and max_tokens and self._is_token_param_error(resp):
            resp = _attempt("max_tokens")
        if not resp.ok:
            self._raise(resp, model, f"{base}/chat/completions")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }

    @staticmethod
    def _is_token_param_error(resp) -> bool:
        if resp.status_code != 400:
            return False
        text = (resp.text or "").lower()
        return "max_tokens" in text or "max_completion_tokens" in text

    @staticmethod
    def _extract_responses_api(data):
        # Prefer the convenience field, else walk the output array.
        text = data.get("output_text")
        if not text:
            parts = []
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") in ("output_text", "text") and block.get("text"):
                            parts.append(block["text"])
            text = "".join(parts)
        usage = data.get("usage", {}) or {}
        return text, {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }

    # ------------------------------------------------------------------
    # Claude (Anthropic Messages API)
    # ------------------------------------------------------------------

    def _claude(self, llm, message, max_tokens, temperature, web_search):
        model = llm.get("identifier")
        key = self._api_key("claude")
        if not key:
            raise RuntimeError("No API key available for provider 'claude'")
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens or 1024,  # required by Anthropic
            "messages": [{"role": "user", "content": message}],
        }
        temp = self._temp(llm, temperature)
        if temp is not None:
            body["temperature"] = temp
        if web_search:
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

        resp = self._post(f"{ANTHROPIC_BASE}/messages", json=body, headers=headers)
        if not resp.ok:
            self._raise(resp, model, f"{ANTHROPIC_BASE}/messages")

        data = resp.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {}) or {}
        return text, {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }

    # ------------------------------------------------------------------
    # Gemini (Google Generative Language API)
    # ------------------------------------------------------------------

    def _gemini(self, llm, message, max_tokens, temperature, web_search):
        model = llm.get("identifier")
        key = self._api_key("gemini")
        if not key:
            raise RuntimeError("No API key available for provider 'gemini'")

        generation_config = {}
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens + GEMINI_TOKEN_BUFFER
        temp = self._temp(llm, temperature)
        if temp is not None:
            generation_config["temperature"] = temp

        body = {"contents": [{"parts": [{"text": message}]}]}
        if generation_config:
            body["generationConfig"] = generation_config
        if web_search:
            body["tools"] = [{"google_search": {}}]

        url = f"{GEMINI_BASE}/models/{model}:generateContent?key={key}"
        resp = self._post(url, json=body, headers={"Content-Type": "application/json"})
        if not resp.ok:
            self._raise(resp, model, f"{GEMINI_BASE}/models/{model}:generateContent")

        data = resp.json()
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {}) or {}
        return text, {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
        }

    # ------------------------------------------------------------------
    # Embeddings (separate from chat completions)
    # ------------------------------------------------------------------

    def embed(self, text: str, provider: str, model: str) -> list[float]:
        """Return a float embedding vector for text using the given provider/model."""
        provider = provider.lower()
        if provider == "openai":
            return self._openai_embed(text, model)
        if provider == "ollama":
            return self._ollama_embed(text, model)
        raise ValueError(f"Unsupported embedding provider: {provider!r}. Use 'openai' or 'ollama'.")

    def _openai_embed(self, text: str, model: str) -> list[float]:
        key = self._api_key("openai")
        if not key:
            raise RuntimeError("No OpenAI API key configured for embeddings.")
        resp = self._post(
            f"{OPENAI_DEFAULT_BASE}/embeddings",
            json={"input": text, "model": model},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if not resp.ok:
            self._raise(resp, model, f"{OPENAI_DEFAULT_BASE}/embeddings")
        return resp.json()["data"][0]["embedding"]

    def _ollama_embed(self, text: str, model: str) -> list[float]:
        url = f"{self.ollama_host.rstrip('/')}/api/embeddings"
        resp = self._post(
            url,
            json={"model": model, "prompt": text},
            headers={"Content-Type": "application/json"},
        )
        if not resp.ok:
            self._raise(resp, model, url)
        return resp.json()["embedding"]

    # ------------------------------------------------------------------
    # Llama (Ollama, local — no web search, no API key)
    # ------------------------------------------------------------------

    def _llama(self, llm, message, max_tokens, temperature):
        model = llm.get("identifier")
        options = {}
        temp = self._temp(llm, temperature)
        if temp is not None:
            options["temperature"] = temp
        if max_tokens:
            options["num_predict"] = max_tokens

        body = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }
        if options:
            body["options"] = options

        url = f"{self.ollama_host.rstrip('/')}/api/chat"
        resp = self._post(url, json=body, headers={"Content-Type": "application/json"})
        if not resp.ok:
            self._raise(resp, model, url)

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return text, {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }
