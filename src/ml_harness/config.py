"""Configuration and path policy. The only module that reads the environment."""
from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass
from pathlib import Path

from .domain.screening import GuardMode

WORKSPACE_ROOT = "/workspace"
# Virtual prefixes the agent sees through its file tools; each maps 1:1 to /workspace/<prefix> in the sandbox.
VIRTUAL_PREFIXES = ("data", "artifacts", "plots")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def raw(self) -> Path: return self.data / "raw"
    @property
    def processed(self) -> Path: return self.data / "processed"
    @property
    def artifacts(self) -> Path: return self.root / "artifacts"
    @property
    def models(self) -> Path: return self.artifacts / "models"
    @property
    def reports(self) -> Path: return self.artifacts / "reports"
    @property
    def logs(self) -> Path: return self.artifacts / "logs"
    @property
    def plots(self) -> Path: return self.root / "plots"
    @property
    def skills(self) -> Path: return self.root / "skills"
    @property
    def docker(self) -> Path: return self.root / "docker"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.models, self.reports, self.logs, self.plots):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SandboxMount:
    host_path: Path
    container_path: str
    read_only: bool = False


@dataclass(frozen=True)
class SandboxPolicy:
    image: str
    timeout_s: int
    memory: str
    cpus: float
    mounts: tuple[SandboxMount, ...]
    network: str = "none"


@dataclass(frozen=True)
class ModelSettings:
    """Provider-neutral LLM configuration.

    provider: 'ollama' | 'openai_compatible' (aliases: vllm, lmstudio, llamacpp) | any LangChain
              init_chat_model provider such as 'anthropic' (hosted).
    """

    provider: str = "anthropic"
    name: str = "claude-sonnet-5-5"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    num_ctx: int = 32768       # Ollama context window; its default (a few K tokens) is far too small for agents
    timeout_s: int = 600       # local generation can be slow


LOCAL_PROVIDERS = {"ollama", "openai_compatible", "vllm", "lmstudio", "llamacpp"}


def parse_model_settings(env: "os._Environ[str] | dict[str, str]") -> ModelSettings:
    """ML_HARNESS_MODEL accepts 'provider:name' (e.g. ollama:qwen3:14b) or, with ML_HARNESS_MODEL_PROVIDER, a bare name."""
    raw = env.get("ML_HARNESS_MODEL", "anthropic:claude-sonnet-5-5")
    provider = env.get("ML_HARNESS_MODEL_PROVIDER")
    if provider:
        name = raw
    elif ":" in raw:
        provider, _, name = raw.partition(":")  # split on the FIRST colon only: ollama:llama3.1:8b
    else:
        raise ValueError("Set ML_HARNESS_MODEL as 'provider:model' or define ML_HARNESS_MODEL_PROVIDER.")
    return ModelSettings(
        provider=provider.strip().lower(),
        name=name.strip(),
        base_url=env.get("ML_HARNESS_MODEL_BASE_URL") or None,
        api_key=env.get("ML_HARNESS_MODEL_API_KEY") or None,
        temperature=float(env.get("ML_HARNESS_MODEL_TEMPERATURE", "0")),
        num_ctx=int(env.get("ML_HARNESS_MODEL_NUM_CTX", "32768")),
        timeout_s=int(env.get("ML_HARNESS_MODEL_TIMEOUT_S", "600")),
    )


@dataclass(frozen=True)
class Settings:
    paths: ProjectPaths
    model: ModelSettings
    sandbox: SandboxPolicy
    recursion_limit: int = 150
    leakage_mode: GuardMode = GuardMode.BLOCK

    @classmethod
    def from_env(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        paths = ProjectPaths(Path(os.environ.get("ML_HARNESS_ROOT", default_root)).resolve())
        mounts = (
            SandboxMount(paths.raw, f"{WORKSPACE_ROOT}/data/raw", read_only=True),  # raw data is immutable
            SandboxMount(paths.processed, f"{WORKSPACE_ROOT}/data/processed"),
            SandboxMount(paths.artifacts, f"{WORKSPACE_ROOT}/artifacts"),
            SandboxMount(paths.plots, f"{WORKSPACE_ROOT}/plots"),
        )
        sandbox = SandboxPolicy(
            image=os.environ.get("ML_HARNESS_SANDBOX_IMAGE", "ml-harness-sandbox:latest"),
            timeout_s=int(os.environ.get("ML_HARNESS_TIMEOUT_S", "600")),
            memory=os.environ.get("ML_HARNESS_SANDBOX_MEMORY", "4g"),
            cpus=float(os.environ.get("ML_HARNESS_SANDBOX_CPUS", "2")),
            mounts=mounts,
        )
        return cls(
            paths=paths,
            model=parse_model_settings(os.environ),
            sandbox=sandbox,
            recursion_limit=int(os.environ.get("ML_HARNESS_RECURSION_LIMIT", "150")),
            leakage_mode=GuardMode(os.environ.get("ML_HARNESS_LEAKAGE_MODE", "block").lower()),
        )


def to_sandbox_path(path: str) -> str:
    """Map an agent-visible path (/data/..., /workspace/data/...) to a validated sandbox path."""
    p = posixpath.normpath(path.strip())
    if not p.startswith("/"):
        raise ValueError(f"Path must be absolute: {path!r}")
    if not p.startswith(WORKSPACE_ROOT + "/"):
        p = WORKSPACE_ROOT + p
    allowed = tuple(f"{WORKSPACE_ROOT}/{prefix}/" for prefix in VIRTUAL_PREFIXES)
    if not p.startswith(allowed):
        raise ValueError(f"Path {path!r} is outside the allowed volumes: {', '.join(VIRTUAL_PREFIXES)}")
    return p
