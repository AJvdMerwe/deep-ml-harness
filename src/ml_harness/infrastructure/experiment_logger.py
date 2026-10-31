from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..domain.interfaces import ExperimentLogger


class JsonlExperimentLogger(ExperimentLogger):
    """Append-only JSON-lines experiment log."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def log(self, record: Mapping[str, Any]) -> str:
        run_id = uuid.uuid4().hex[:8]
        entry = {"run_id": run_id, "logged_at": datetime.now(timezone.utc).isoformat(), **record}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return run_id
