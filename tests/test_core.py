import json
import subprocess

import pytest

from ml_harness.config import to_sandbox_path
from ml_harness.domain.models import ExecutionResult
from ml_harness.infrastructure.docker_executor import DockerExecutor
from ml_harness.infrastructure.experiment_logger import JsonlExperimentLogger
from ml_harness.tools.code_execution import CodeExecutionTools
from ml_harness.tools.data_catalog import DataCatalogTools
from ml_harness.tools.experiment_log import ExperimentLogTools


def test_render_clips_long_output():
    out = ExecutionResult(0, stdout="x" * 20000).render(max_chars=1000)
    assert "truncated" in out and len(out) < 1500


def test_sandbox_path_mapping_and_guard():
    assert to_sandbox_path("/data/raw/a.csv") == "/workspace/data/raw/a.csv"
    assert to_sandbox_path("/workspace/plots/x.png") == "/workspace/plots/x.png"
    for bad in ("/etc/passwd", "/data/../etc/passwd", "relative.csv", "/src/x.py"):
        with pytest.raises(ValueError):
            to_sandbox_path(bad)


def test_docker_command_is_locked_down(settings):
    cmd = DockerExecutor(settings.sandbox).build_command("c1")
    joined = " ".join(cmd)
    assert "--network none" in joined and "--cap-drop ALL" in joined and "--read-only" in cmd
    assert "--privileged" not in cmd
    assert any(c.endswith("/workspace/data/raw:ro") for c in cmd)  # raw data mounted read-only
    assert cmd[-3:] == ["python", "-u", "-"]


def test_docker_execute_and_timeout(settings, monkeypatch):
    ex = DockerExecutor(settings.sandbox)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "hi", ""))
    assert ex.execute("print('hi')").ok

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "kill" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise subprocess.TimeoutExpired(cmd, 1, output=b"partial")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = ex.execute("while True: pass", timeout_s=1)
    assert res.timed_out and res.stdout == "partial"
    assert any("kill" in c for c in calls)  # container is reaped


def test_run_python_tool_delegates(fake_executor):
    tool = CodeExecutionTools(fake_executor).tools()[0]
    assert "fake-ok" in tool.invoke({"code": "print(1)"})
    assert fake_executor.calls == ["print(1)"]


def test_catalog_lists_and_profiles(paths, fake_executor):
    (paths.raw / "churn.csv").write_text("a,b\n1,2\n")
    tools = {t.name: t for t in DataCatalogTools(paths, fake_executor).tools()}
    assert "/data/raw/churn.csv" in tools["list_datasets"].invoke({})
    assert tools["profile_dataset"].invoke({"path": "/etc/passwd"}).startswith("ERROR")
    tools["profile_dataset"].invoke({"path": "/data/raw/churn.csv", "target": "b"})
    assert "/workspace/data/raw/churn.csv" in fake_executor.calls[0]


def test_experiment_log_is_jsonl(paths):
    log_path = paths.logs / "experiments.jsonl"
    tool = ExperimentLogTools(JsonlExperimentLogger(log_path)).tools()[0]
    tool.invoke({"run_name": "r1", "model_type": "xgb", "params": {"max_depth": 4}, "metrics": {"auc": 0.8}})
    rec = json.loads(log_path.read_text().splitlines()[0])
    assert rec["metrics"]["auc"] == 0.8 and rec["run_id"]
