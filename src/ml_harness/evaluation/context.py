from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.screening import ScreeningContext, ScreeningContextProvider


def _cols(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    if not isinstance(value, (list, tuple)):
        return frozenset()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict):
            col = item.get("column") or item.get("name")
            if isinstance(col, str):
                out.add(col)
    return frozenset(out)


class EdaSummaryContextProvider(ScreeningContextProvider):
    """Reads target / leakage_suspects / dropped_columns from the EDA stage's summary JSON.

    Re-read on every screening so the guard always sees the latest EDA output. A missing or
    malformed file yields an empty context (the guard then relies on in-script inference only).
    """

    def __init__(self, summary_path: Path) -> None:
        self._path = summary_path

    def load(self) -> ScreeningContext:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ScreeningContext()
        if not isinstance(data, dict):
            return ScreeningContext()
        return ScreeningContext(
            target_columns=_cols(data.get("target")),
            suspect_columns=_cols(data.get("leakage_suspects")),
            dropped_columns=_cols(data.get("dropped_columns")),
        )
