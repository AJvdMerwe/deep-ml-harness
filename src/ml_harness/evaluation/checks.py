"""Leakage rules LK001-LK006. Static analysis only: fast, deterministic, no data access."""
from __future__ import annotations

import ast

from ..domain.screening import Finding, ScriptCheck, ScriptView, Severity
from .ast_utils import (X_NAME, Y_NAME, call_name, is_test_name, lhs_name, names_in, str_consts)

_PASSTHROUGH = {"copy", "select_dtypes", "reset_index", "fillna", "dropna", "sample", "astype", "head"}
_PREPROCESSORS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "PowerTransformer", "QuantileTransformer",
    "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "SimpleImputer", "KNNImputer", "IterativeImputer",
    "PCA", "SelectKBest", "SelectFromModel", "VarianceThreshold", "ColumnTransformer", "TfidfVectorizer",
    "CountVectorizer", "KBinsDiscretizer",
}
_SCORERS = {"f1_score", "precision_score", "recall_score", "accuracy_score", "fbeta_score"}


def _unwrap_frame(node: ast.AST):
    """Walk a method/subscript chain down to its base name -> (base, dropped_consts, selected_consts|None)."""
    dropped: set[str] = set()
    selected: set[str] | None = None
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "drop":
                dropped |= str_consts(node)
            elif node.func.attr not in _PASSTHROUGH:
                return None
            node = node.func.value
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, (ast.List, ast.Tuple, ast.Constant, ast.Name)):
                selected = (selected or set()) | str_consts(node.slice)  # explicit column selection
            node = node.value  # boolean masks etc. keep all columns
        elif isinstance(node, ast.Name):
            return node.id, dropped, selected
        else:
            return None


def _scrub_events(tree: ast.AST, targets: frozenset[str]) -> list[tuple[int, str]]:
    """(lineno, frame) pairs where a target column is removed from a frame in place or by rebinding."""
    events: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            frame, attr = node.func.value.id, node.func.attr
            if attr == "pop" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in targets:
                events.append((node.lineno, frame))
            elif attr == "drop" and str_consts(node) & targets and any(
                    k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                    for k in node.keywords):
                events.append((node.lineno, frame))
        elif isinstance(node, ast.Assign):
            name, unwrapped = lhs_name(node), _unwrap_frame(node.value)
            if name and unwrapped and unwrapped[1] & targets:
                events += [(node.lineno, name), (node.lineno, unwrapped[0])]
    return events


class TargetInFeaturesCheck(ScriptCheck):
    """LK001: the feature matrix still contains the label column."""

    rule_id = "LK001"

    def check(self, view: ScriptView) -> list[Finding]:
        if not view.targets:
            return []
        scrubs = _scrub_events(view.tree, view.targets)
        out: list[Finding] = []
        for node in ast.walk(view.tree):
            if not isinstance(node, ast.Assign):
                continue
            name = lhs_name(node)
            if not name or not X_NAME.match(name) or Y_NAME.match(name):
                continue
            unwrapped = _unwrap_frame(node.value)
            if not unwrapped:
                continue
            base, dropped, selected = unwrapped
            if base not in view.label_refs:
                continue
            at_risk = set(view.label_refs[base]) | set(view.targets & view.context.target_columns)
            if selected is not None:
                leaked = at_risk & selected
            elif any(f == base and ln < node.lineno for ln, f in scrubs):
                leaked = set()
            else:
                leaked = at_risk - dropped
            if leaked:
                out.append(Finding(
                    self.rule_id, Severity.BLOCK, line=node.lineno,
                    message=(f"Feature matrix '{name}' is built from '{base}' but still contains target column(s) "
                             f"{sorted(leaked)}. Use y = {base}.pop('<target>') or {base}.drop(columns=['<target>']) "
                             "BEFORE building X."),
                ))
        return out


class FitBeforeSplitCheck(ScriptCheck):
    """LK002: preprocessing/feature selection fit on the full data before train_test_split."""

    rule_id = "LK002"

    def check(self, view: ScriptView) -> list[Finding]:
        split_lines = [n.lineno for n in ast.walk(view.tree)
                       if isinstance(n, ast.Call) and call_name(n) == "train_test_split"]
        if not split_lines:
            return []
        first_split = min(split_lines)
        proc_vars: dict[str, str] = {}
        for node in ast.walk(view.tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = lhs_name(node)
                if name and call_name(node.value) in _PREPROCESSORS:
                    proc_vars[name] = call_name(node.value)
        out: list[Finding] = []
        for node in ast.walk(view.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"fit", "fit_transform"} and node.lineno < first_split):
                continue
            recv = node.func.value
            cls = proc_vars.get(recv.id) if isinstance(recv, ast.Name) else (
                call_name(recv) if isinstance(recv, ast.Call) and call_name(recv) in _PREPROCESSORS else None)
            if cls:
                out.append(Finding(
                    self.rule_id, Severity.BLOCK, line=node.lineno,
                    message=(f"{cls}.{node.func.attr}() runs before train_test_split (line {first_split}); its "
                             "statistics would include test rows. Split first, fit on the training split only."),
                ))
        return out


class TargetDerivedFeatureCheck(ScriptCheck):
    """LK003: a new feature column is computed from the label (e.g. target encoding on the full frame)."""

    rule_id = "LK003"
    waivable = True

    @staticmethod
    def _refs_target(node: ast.AST, targets: frozenset[str]) -> bool:
        return any(isinstance(s, ast.Subscript) and isinstance(s.slice, ast.Constant) and s.slice.value in targets
                   for s in ast.walk(node))

    def check(self, view: ScriptView) -> list[Finding]:
        if not view.targets:
            return []
        out: list[Finding] = []

        def flag(line: int, col: str) -> None:
            out.append(Finding(
                self.rule_id, Severity.BLOCK, line=line, waivable=True,
                message=(f"Feature '{col}' is derived from target column(s) {sorted(view.targets)}. Compute "
                         "target-based statistics from the TRAINING split only (e.g. from y_train), or overwrite "
                         "the target column itself instead of creating a second copy.")))

        for node in ast.walk(view.tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str)
                            and t.slice.value not in view.targets and not Y_NAME.match(t.value.id)
                            and self._refs_target(node.value, view.targets)):
                        flag(node.lineno, t.slice.value)
            elif isinstance(node, ast.Call) and call_name(node) == "assign":
                for kw in node.keywords:
                    if kw.arg and kw.arg not in view.targets and self._refs_target(kw.value, view.targets):
                        flag(node.lineno, kw.arg)
        return out


class EvalOnTestCheck(ScriptCheck):
    """LK004: test data used to fit, early-stop or tune."""

    rule_id = "LK004"

    def check(self, view: ScriptView) -> list[Finding]:
        out: list[Finding] = []
        for node in ast.walk(view.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"fit", "fit_transform"}):
                continue
            for kw in node.keywords:
                if kw.arg == "eval_set" and any(is_test_name(n) for n in names_in(kw.value)):
                    out.append(Finding(self.rule_id, Severity.BLOCK, line=node.lineno,
                                       message="eval_set contains test data: early stopping would tune on the test "
                                               "set. Use a validation split."))
            if node.args and any(is_test_name(n) for n in names_in(node.args[0])):
                out.append(Finding(self.rule_id, Severity.BLOCK, line=node.lineno,
                                   message=f".{node.func.attr}() is called on test data. Fit only on the training "
                                           "split; use transform()/predict() on test."))
        return out


class ThresholdOnTestCheck(ScriptCheck):
    """LK005 (warning): decision threshold or hyperparameters selected using test labels."""

    rule_id = "LK005"
    waivable = True

    def check(self, view: ScriptView) -> list[Finding]:
        out: list[Finding] = []
        selects = any(w in view.source for w in ("argmax", "argmin"))
        for node in ast.walk(view.tree):
            if (isinstance(node, ast.Call) and call_name(node) in {"precision_recall_curve", "roc_curve"}
                    and selects and node.args and any(is_test_name(n) for n in names_in(node.args[0]))):
                out.append(Finding(self.rule_id, Severity.WARN, line=node.lineno, waivable=True,
                                   message=f"{call_name(node)}() on test labels plus argmax/argmin looks like "
                                           "threshold selection on the test set. Select on the validation split."))
            elif isinstance(node, (ast.For, ast.While)):
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call) and call_name(sub) in _SCORERS and sub.args
                            and any(is_test_name(n) for n in names_in(sub.args[0]))):
                        out.append(Finding(self.rule_id, Severity.WARN, line=sub.lineno, waivable=True,
                                           message=f"{call_name(sub)}() on test labels inside a loop: if this picks a "
                                                   "threshold or hyperparameter, use the validation split."))
                        break
        return out


class SuspectColumnCheck(ScriptCheck):
    """LK006: a column the EDA stage flagged as a leakage suspect is neither dropped nor excluded."""

    rule_id = "LK006"
    waivable = True

    def check(self, view: ScriptView) -> list[Finding]:
        suspects = view.context.suspect_columns - view.context.dropped_columns - view.targets
        if not suspects:
            return []
        dropped: set[str] = set()
        selected: set[str] = set()
        for node in ast.walk(view.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"drop", "pop"}:
                dropped |= str_consts(node)
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.List):
                selected |= str_consts(node.slice)
        return [
            Finding(self.rule_id, Severity.BLOCK, waivable=True,
                    message=(f"EDA flagged '{col}' as a leakage suspect but the script never drops it. Drop it "
                             "explicitly, or justify keeping it with a waiver comment."))
            for col in sorted(suspects)
            if col not in dropped and not (selected and col not in selected)
        ]


def default_checks() -> list[ScriptCheck]:
    return [TargetInFeaturesCheck(), FitBeforeSplitCheck(), TargetDerivedFeatureCheck(),
            EvalOnTestCheck(), ThresholdOnTestCheck(), SuspectColumnCheck()]
