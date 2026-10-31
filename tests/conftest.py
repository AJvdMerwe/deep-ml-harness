import pytest

from ml_harness.config import ModelSettings, ProjectPaths, SandboxPolicy, Settings, SandboxMount
from ml_harness.domain.interfaces import CodeExecutor
from ml_harness.domain.models import ExecutionResult


class FakeExecutor(CodeExecutor):
    def __init__(self):
        self.calls = []

    def execute(self, code, *, timeout_s=None):
        self.calls.append(code)
        return ExecutionResult(exit_code=0, stdout="fake-ok")


@pytest.fixture
def fake_executor():
    return FakeExecutor()


@pytest.fixture
def paths(tmp_path):
    p = ProjectPaths(tmp_path)
    p.ensure()
    (p.skills).mkdir(exist_ok=True)
    return p


@pytest.fixture
def settings(paths):
    policy = SandboxPolicy(
        image="img:test", timeout_s=5, memory="1g", cpus=1.0,
        mounts=(SandboxMount(paths.raw, "/workspace/data/raw", True), SandboxMount(paths.plots, "/workspace/plots")),
    )
    return Settings(paths=paths, model=ModelSettings(provider="ollama", name="qwen3:14b"), sandbox=policy)
