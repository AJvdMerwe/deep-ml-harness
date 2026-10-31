import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml_harness.config import ModelSettings, parse_model_settings
from ml_harness.domain.errors import ModelUnavailableError
from ml_harness.infrastructure.model_factories import (HostedModelFactory, OllamaModelFactory,
                                                       OpenAICompatibleModelFactory, create_model_factory)


@pytest.mark.parametrize("env,provider,name", [
    ({"ML_HARNESS_MODEL": "anthropic:claude-sonnet-5-5"}, "anthropic", "claude-sonnet-5-5"),
    ({"ML_HARNESS_MODEL": "ollama:llama3.1:8b"}, "ollama", "llama3.1:8b"),            # first colon only
    ({"ML_HARNESS_MODEL": "qwen3:14b", "ML_HARNESS_MODEL_PROVIDER": "ollama"}, "ollama", "qwen3:14b"),
    ({"ML_HARNESS_MODEL": "Qwen/Qwen3-30B", "ML_HARNESS_MODEL_PROVIDER": "VLLM"}, "vllm", "Qwen/Qwen3-30B"),
])
def test_model_env_parsing(env, provider, name):
    m = parse_model_settings(env)
    assert (m.provider, m.name) == (provider, name)


def test_bare_name_without_provider_is_rejected():
    with pytest.raises(ValueError):
        parse_model_settings({"ML_HARNESS_MODEL": "qwen3"})


def test_env_overrides():
    m = parse_model_settings({"ML_HARNESS_MODEL": "ollama:x", "ML_HARNESS_MODEL_BASE_URL": "http://gpu-box:11434",
                              "ML_HARNESS_MODEL_NUM_CTX": "65536", "ML_HARNESS_MODEL_TEMPERATURE": "0.2"})
    assert m.base_url == "http://gpu-box:11434" and m.num_ctx == 65536 and m.temperature == 0.2


@pytest.mark.parametrize("provider,cls", [("ollama", OllamaModelFactory), ("vllm", OpenAICompatibleModelFactory),
                                          ("lmstudio", OpenAICompatibleModelFactory),
                                          ("llamacpp", OpenAICompatibleModelFactory),
                                          ("openai_compatible", OpenAICompatibleModelFactory),
                                          ("anthropic", HostedModelFactory)])
def test_factory_selection(provider, cls):
    assert isinstance(create_model_factory(ModelSettings(provider=provider, name="m")), cls)


def test_hosted_factory_returns_provider_string():
    assert create_model_factory(ModelSettings(provider="anthropic", name="claude-x")).create() == "anthropic:claude-x"


def test_ollama_model_is_configured():
    pytest.importorskip("langchain_ollama")
    m = create_model_factory(ModelSettings(provider="ollama", name="qwen3:14b", num_ctx=16384,
                                           base_url="http://h:1")).create()
    assert m.model == "qwen3:14b" and m.num_ctx == 16384 and m.base_url == "http://h:1"


def test_openai_compatible_model_is_configured():
    pytest.importorskip("langchain_openai")
    m = create_model_factory(ModelSettings(provider="vllm", name="qwen", base_url="http://h:8000/v1")).create()
    assert m.model_name == "qwen" and m.openai_api_base == "http://h:8000/v1"
    assert m.openai_api_key.get_secret_value() == "not-needed"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {"/api/tags": {"models": [{"name": "qwen3:14b"}, {"name": "llama3.1:latest"}]},
                  "/v1/models": {"data": [{"id": "served-model"}]}}
        body = bodies.get(self.path)
        self.send_response(200 if body else 404)
        self.end_headers()
        if body:
            self.wfile.write(json.dumps(body).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_health_check(server):
    ok = lambda n: OllamaModelFactory(ModelSettings(provider="ollama", name=n, base_url=server)).health_check()
    ok("qwen3:14b")
    ok("llama3.1")  # resolves to llama3.1:latest
    with pytest.raises(ModelUnavailableError, match="ollama pull"):
        ok("missing-model")


def test_openai_compatible_health_check(server):
    mk = lambda n: OpenAICompatibleModelFactory(ModelSettings(provider="vllm", name=n, base_url=f"{server}/v1"))
    mk("served-model").health_check()
    with pytest.raises(ModelUnavailableError, match="served-model"):
        mk("other").health_check()


def test_unreachable_server_gives_actionable_error():
    f = OllamaModelFactory(ModelSettings(provider="ollama", name="x", base_url="http://127.0.0.1:1"))
    with pytest.raises(ModelUnavailableError, match="ollama serve"):
        f.health_check()
