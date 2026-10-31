import pytest

pytest.importorskip("deepagents")

from ml_harness.agents.subagents import EdaSubAgentProvider, TrainingSubAgentProvider
from ml_harness.app.composition import build_agent
from ml_harness.tools.code_execution import CodeExecutionTools


def test_subagent_specs_are_scoped(fake_executor):
    tools = CodeExecutionTools(fake_executor)
    eda, train = EdaSubAgentProvider([tools]).build(), TrainingSubAgentProvider([tools]).build()
    assert eda["name"] == "data-eda" and train["name"] == "ml-training"
    assert eda["skills"] != train["skills"]
    assert "/skills/eda/" in eda["skills"] and "/skills/training/" in train["skills"]


def test_agent_builds_on_local_model(settings, fake_executor):
    pytest.importorskip("langchain_ollama")
    assert build_agent(settings, fake_executor) is not None


def test_training_agent_gets_guard_and_eda_does_not(settings, fake_executor, monkeypatch):
    from ml_harness.domain.screening import GuardMode
    from ml_harness.evaluation.middleware import LeakageGuardMiddleware
    from dataclasses import replace

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # composition attaches the guard only to ml-training; verify via the provider contract
    guard = LeakageGuardMiddleware.__new__(LeakageGuardMiddleware)
    train = TrainingSubAgentProvider([CodeExecutionTools(fake_executor)], middleware=[guard]).build()
    eda = EdaSubAgentProvider([CodeExecutionTools(fake_executor)]).build()
    assert train["middleware"] == [guard] and "middleware" not in eda
    assert build_agent(replace(settings, leakage_mode=GuardMode.OFF), fake_executor) is not None
