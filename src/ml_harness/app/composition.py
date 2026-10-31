"""Composition root: the ONE place where concrete classes are chosen and wired together."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from ..agents.root import ResearchAgentBuilder
from ..agents.subagents import EdaSubAgentProvider, TrainingSubAgentProvider
from ..config import Settings
from ..domain.interfaces import CodeExecutor
from ..domain.screening import GuardMode
from ..evaluation.checks import default_checks
from ..evaluation.context import EdaSummaryContextProvider
from ..evaluation.middleware import LeakageGuardMiddleware
from ..evaluation.screener import LeakageScreener
from ..infrastructure.docker_executor import DockerExecutor
from ..infrastructure.experiment_logger import JsonlExperimentLogger
from ..infrastructure.model_factories import create_model_factory
from ..infrastructure.workspace_backend import WorkspaceBackendBuilder
from ..tools.code_execution import CodeExecutionTools
from ..tools.data_catalog import DataCatalogTools
from ..tools.experiment_log import ExperimentLogTools


def build_agent(settings: Settings, executor: CodeExecutor | None = None, model=None):
    executor = executor or DockerExecutor(settings.sandbox)
    model = model if model is not None else create_model_factory(settings.model).create()

    code_tools = CodeExecutionTools(executor)
    catalog_tools = DataCatalogTools(settings.paths, executor)
    log_tools = ExperimentLogTools(JsonlExperimentLogger(settings.paths.logs / "experiments.jsonl"))

    guard = LeakageGuardMiddleware(
        LeakageScreener(default_checks(), EdaSummaryContextProvider(settings.paths.reports / "eda_summary.json")),
        mode=settings.leakage_mode,
        audit=JsonlExperimentLogger(settings.paths.logs / "leakage_audit.jsonl"),
    )
    training_middleware = [] if settings.leakage_mode is GuardMode.OFF else [guard]

    return ResearchAgentBuilder(
        model=model,
        backend=WorkspaceBackendBuilder(settings.paths).build(),
        tool_providers=[catalog_tools],  # root only discovers data; heavy work is delegated
        subagent_providers=[
            EdaSubAgentProvider([code_tools, catalog_tools]),
            TrainingSubAgentProvider([code_tools, catalog_tools, log_tools], middleware=training_middleware),
        ],
        checkpointer=MemorySaver(),
        recursion_limit=settings.recursion_limit,
    ).build()
