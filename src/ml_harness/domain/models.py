from __future__ import annotations

from dataclasses import dataclass


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars truncated] ...\n{text[-half:]}"


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self, max_chars: int = 8000) -> str:
        """LLM-friendly rendering; head+tail clipping keeps tracebacks and final prints."""
        status = f"exit_code={self.exit_code} duration={self.duration_s:.1f}s"
        if self.timed_out:
            status += " TIMED_OUT"
        parts = [status]
        if self.stdout.strip():
            parts.append("--- stdout ---\n" + _clip(self.stdout, max_chars))
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + _clip(self.stderr, max_chars))
        return "\n".join(parts)
