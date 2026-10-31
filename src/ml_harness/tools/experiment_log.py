from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ..domain.interfaces import ExperimentLogger, ToolProvider


class ExperimentLogTools(ToolProvider):
    def __init__(self, logger: ExperimentLogger) -> None:
        self._logger = logger

    def tools(self) -> list[BaseTool]:
        return [StructuredTool.from_function(self.log_experiment)]

    def log_experiment(
        self,
        run_name: str,
        model_type: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifact_paths: list[str] | None = None,
    ) -> str:
        """Append an experiment record (hyperparameters, metrics, artifact paths) to artifacts/logs/experiments.jsonl.

        Call once per hyperparameter configuration worth keeping and once for the final model.
        """
        run_id = self._logger.log(
            {
                "run_name": run_name,
                "model_type": model_type,
                "params": params,
                "metrics": metrics,
                "artifact_paths": artifact_paths or [],
            }
        )
        return f"logged run_id={run_id}"
