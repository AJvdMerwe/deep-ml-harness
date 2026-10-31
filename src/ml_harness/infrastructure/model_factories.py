"""Chat-model adapters. Add a provider by adding a class and one registry entry (open/closed)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from langchain_core.language_models import BaseChatModel

from ..config import ModelSettings
from ..domain.errors import ModelUnavailableError
from ..domain.interfaces import ChatModelFactory


def _get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-configured local URL
        return json.loads(resp.read().decode("utf-8"))


def _import_hint(extra: str, package: str) -> ModelUnavailableError:
    return ModelUnavailableError(f"Package '{package}' is not installed. Run: pip install -e \".[{extra}]\"")


class HostedModelFactory(ChatModelFactory):
    """Hosted providers via LangChain's 'provider:model' strings (anthropic, openai, google_genai, ...)."""

    def __init__(self, settings: ModelSettings) -> None:
        self._s = settings

    def create(self) -> str:
        return f"{self._s.provider}:{self._s.name}"

    @property
    def description(self) -> str:
        return f"hosted {self._s.provider}:{self._s.name}"


class OllamaModelFactory(ChatModelFactory):
    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, settings: ModelSettings) -> None:
        self._s = settings
        self._base = (settings.base_url or self.DEFAULT_URL).rstrip("/")

    def create(self) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise _import_hint("ollama", "langchain-ollama") from exc
        return ChatOllama(
            model=self._s.name,
            base_url=self._base,
            temperature=self._s.temperature,
            num_ctx=self._s.num_ctx,
            client_kwargs={"timeout": self._s.timeout_s},
        )

    def health_check(self) -> None:
        try:
            tags = _get_json(f"{self._base}/api/tags")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelUnavailableError(
                f"Cannot reach Ollama at {self._base} ({exc}). Start it with 'ollama serve' "
                "or set ML_HARNESS_MODEL_BASE_URL."
            ) from exc
        installed = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self._s.name
        if wanted not in installed and f"{wanted}:latest" not in installed:
            raise ModelUnavailableError(
                f"Model '{wanted}' is not pulled in Ollama. Run: ollama pull {wanted}\n"
                f"Installed: {', '.join(sorted(installed)) or 'none'}"
            )

    @property
    def description(self) -> str:
        return f"ollama {self._s.name} @ {self._base} (num_ctx={self._s.num_ctx})"


class OpenAICompatibleModelFactory(ChatModelFactory):
    """vLLM, LM Studio, llama.cpp server, LocalAI, TGI... anything exposing /v1/chat/completions."""

    DEFAULT_URL = "http://localhost:8000/v1"  # vLLM; LM Studio: :1234/v1, llama.cpp: :8080/v1

    def __init__(self, settings: ModelSettings) -> None:
        self._s = settings
        self._base = (settings.base_url or self.DEFAULT_URL).rstrip("/")

    def create(self) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise _import_hint("openai", "langchain-openai") from exc
        return ChatOpenAI(
            model=self._s.name,
            base_url=self._base,
            api_key=self._s.api_key or "not-needed",  # local servers usually ignore the key
            temperature=self._s.temperature,
            timeout=self._s.timeout_s,
            max_retries=2,
        )

    def health_check(self) -> None:
        headers = {"Authorization": f"Bearer {self._s.api_key}"} if self._s.api_key else None
        try:
            models = _get_json(f"{self._base}/models", headers)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ModelUnavailableError(
                f"Cannot reach an OpenAI-compatible server at {self._base} ({exc}). Check the server is running "
                "and ML_HARNESS_MODEL_BASE_URL ends with /v1."
            ) from exc
        ids = {m.get("id", "") for m in models.get("data", [])}
        if ids and self._s.name not in ids:
            raise ModelUnavailableError(
                f"Server does not list model '{self._s.name}'. Available: {', '.join(sorted(ids))}"
            )

    @property
    def description(self) -> str:
        return f"openai-compatible {self._s.name} @ {self._base}"


_REGISTRY: dict[str, Callable[[ModelSettings], ChatModelFactory]] = {
    "ollama": OllamaModelFactory,
    "openai_compatible": OpenAICompatibleModelFactory,
    "vllm": OpenAICompatibleModelFactory,
    "lmstudio": OpenAICompatibleModelFactory,
    "llamacpp": OpenAICompatibleModelFactory,
}


def create_model_factory(settings: ModelSettings) -> ChatModelFactory:
    return _REGISTRY.get(settings.provider, HostedModelFactory)(settings)
