from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool

from ..config import WORKSPACE_ROOT, ProjectPaths, to_sandbox_path
from ..domain.interfaces import CodeExecutor, ToolProvider

_DATA_SUFFIXES = {".csv", ".parquet", ".json", ".feather"}

_PROFILE_SCRIPT = """
import json, pathlib
import pandas as pd
args = json.loads(__ARGS__)
p = pathlib.Path(args["path"])
readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".json": pd.read_json, ".feather": pd.read_feather}
df = readers[p.suffix.lower()](p)
print("shape:", df.shape)
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "null_pct": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(),
})
print(summary.to_string())
print("duplicate_rows:", int(df.duplicated().sum()))
print(df.head(5).to_string())
t = args.get("target")
if t and t in df.columns:
    print("target distribution:")
    print(df[t].value_counts(normalize=True, dropna=False).round(4).to_string())
"""


class DataCatalogTools(ToolProvider):
    """Dataset discovery and fast profiling (profiling itself runs inside the sandbox)."""

    def __init__(self, paths: ProjectPaths, executor: CodeExecutor) -> None:
        self._paths = paths
        self._executor = executor

    def tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(self.list_datasets),
            StructuredTool.from_function(self.profile_dataset),
        ]

    def list_datasets(self) -> str:
        """List available datasets (csv/parquet/json/feather) in data/raw and data/processed with sizes."""
        lines: list[str] = []
        for base in (self._paths.raw, self._paths.processed):
            for f in sorted(base.rglob("*")):
                if f.is_file() and f.suffix.lower() in _DATA_SUFFIXES:
                    rel = f.relative_to(self._paths.data).as_posix()
                    lines.append(f"/data/{rel}  ({f.stat().st_size / 1024:.1f} KiB)  sandbox: {WORKSPACE_ROOT}/data/{rel}")
        return "\n".join(lines) or "No datasets found. Place files in data/raw/."

    def profile_dataset(self, path: str, target: str | None = None) -> str:
        """Profile a dataset: shape, dtypes, null %, unique counts, duplicates, head and optional target balance.

        Args:
            path: dataset path, e.g. /data/raw/churn.csv
            target: optional target column name
        """
        try:
            sandbox_path = to_sandbox_path(path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        args = json.dumps({"path": sandbox_path, "target": target})
        script = _PROFILE_SCRIPT.replace("__ARGS__", json.dumps(args))
        return self._executor.execute(script).render()
