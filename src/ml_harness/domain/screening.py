"""Ports and value objects for pre-execution script screening."""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class Severity(str, Enum):
    BLOCK = "block"  # script must not run
    WARN = "warn"    # script runs; finding is appended to the tool output
    INFO = "info"    # e.g. a waived finding, kept for transparency


class GuardMode(str, Enum):
    BLOCK = "block"  # block scripts that have BLOCK findings
    WARN = "warn"    # never block; surface every finding as a warning
    OFF = "off"      # guard not attached


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    message: str
    line: int | None = None
    waivable: bool = False


@dataclass(frozen=True)
class ScreeningContext:
    """Facts about the dataset gathered by earlier pipeline stages (e.g. the EDA summary)."""

    target_columns: frozenset[str] = frozenset()
    suspect_columns: frozenset[str] = frozenset()
    dropped_columns: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ScriptView:
    """Everything a check may look at. Checks never touch the filesystem or the sandbox."""

    source: str
    tree: ast.Module
    targets: frozenset[str]                         # context targets + targets inferred from the script
    label_refs: Mapping[str, frozenset[str]]        # dataframe name -> target columns its label was taken from
    context: ScreeningContext


@dataclass(frozen=True)
class ScreeningReport:
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.BLOCK)

    @property
    def blocked(self) -> bool:
        return bool(self.blocking)

    @staticmethod
    def _fmt(f: Finding) -> str:
        where = f" line {f.line}" if f.line else ""
        return f"[{f.severity.value.upper()} {f.rule_id}{where}] {f.message}"

    def render_block(self) -> str:
        lines = ["LEAKAGE GUARD: script was NOT executed. Fix the findings below and resubmit.", ""]
        lines += [self._fmt(f) for f in self.findings]
        if any(f.waivable and f.severity is Severity.BLOCK for f in self.findings):
            lines += [
                "",
                "Waivable rules only: if a finding is a verified false positive, add a comment "
                "'# leakage-waive: <RULE_ID> <reason of at least 10 characters>'. Waivers are audited.",
            ]
        return "\n".join(lines)

    def render_notes(self) -> str:
        return "LEAKAGE GUARD notes (script executed):\n" + "\n".join(self._fmt(f) for f in self.findings)


class ScriptCheck(ABC):
    """One leakage rule (Strategy). Add a rule by adding a class; nothing else changes."""

    rule_id: str
    waivable: bool = False

    @abstractmethod
    def check(self, view: ScriptView) -> list[Finding]: ...


class ScreeningContextProvider(ABC):
    @abstractmethod
    def load(self) -> ScreeningContext: ...


class ScriptScreener(ABC):
    @abstractmethod
    def screen(self, code: str) -> ScreeningReport: ...
