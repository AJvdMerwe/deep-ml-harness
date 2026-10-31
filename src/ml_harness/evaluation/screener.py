from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import Sequence

from ..domain.screening import (Finding, ScreeningContextProvider, ScreeningReport, ScriptCheck, ScriptScreener,
                                ScriptView, Severity)
from .ast_utils import infer_label_refs

# Only model-building scripts are screened; pure EDA/plotting scripts legitimately read the target.
_TRAINING_HINT = re.compile(r"\.fit\(|\.fit_transform\(|train_test_split|XGB\w*\(|cross_val|SearchCV|\.train\(")
_WAIVER = re.compile(r"#\s*leakage-waive:\s*(LK\d{3})\s+(\S.{9,})")


def parse_waivers(code: str) -> dict[str, str]:
    waivers: dict[str, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT and (m := _WAIVER.search(tok.string)):
                waivers[m.group(1)] = m.group(2).strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return waivers


class LeakageScreener(ScriptScreener):
    """Runs a set of ScriptChecks over a script. Depends on abstractions only (checks + context provider)."""

    def __init__(self, checks: Sequence[ScriptCheck], context_provider: ScreeningContextProvider) -> None:
        self._checks = tuple(checks)
        self._context_provider = context_provider

    def screen(self, code: str) -> ScreeningReport:
        if not _TRAINING_HINT.search(code):
            return ScreeningReport()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ScreeningReport()  # let the sandbox report the SyntaxError; nothing to analyse

        context = self._context_provider.load()
        label_refs = infer_label_refs(tree, context.target_columns)
        targets = frozenset(context.target_columns | {c for cols in label_refs.values() for c in cols})
        view = ScriptView(source=code, tree=tree, targets=targets, label_refs=label_refs, context=context)

        findings = [f for check in self._checks for f in check.check(view)]
        findings = self._apply_waivers(findings, parse_waivers(code))
        findings.sort(key=lambda f: (f.line or 0, f.rule_id))
        return ScreeningReport(tuple(findings))

    @staticmethod
    def _apply_waivers(findings: list[Finding], waivers: dict[str, str]) -> list[Finding]:
        out: list[Finding] = []
        for f in findings:
            if f.waivable and f.rule_id in waivers:
                out.append(Finding(f.rule_id, Severity.INFO, line=f.line,
                                   message=f"WAIVED ({waivers[f.rule_id]}): {f.message}"))
            else:
                out.append(f)
        return out
