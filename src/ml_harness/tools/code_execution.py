from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from ..domain.interfaces import CodeExecutor, ToolProvider


class CodeExecutionTools(ToolProvider):
    """Single responsibility: expose sandboxed Python execution to agents."""

    def __init__(self, executor: CodeExecutor) -> None:
        self._executor = executor

    def tools(self) -> list[BaseTool]:
        executor = self._executor

        def run_python(code: str) -> str:
            """Execute a complete, self-contained Python script in an isolated sandbox and return stdout/stderr.

            Each call is a fresh process: no variables persist between calls, so re-load data every time.
            Pre-installed: pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn, joblib, optuna,
            shap, onnxmltools, skl2onnx, onnxruntime, pyarrow. There is no network access.

            Filesystem (sandbox paths):
              /workspace/data/raw/        read-only input data
              /workspace/data/processed/  cleaned datasets (parquet preferred)
              /workspace/artifacts/{models,reports,logs}/  models (.pkl/.onnx), JSON/MD reports, logs
              /workspace/plots/           PNG figures (call plt.savefig; never plt.show)
            Always print the results you need to see; only printed output is returned.
            """
            return executor.execute(code).render()

        return [StructuredTool.from_function(run_python)]
