from __future__ import annotations

import ast
import re
from collections import defaultdict

_TEST_NAME = re.compile(r"(?i)(^|_)(test|te|holdout|hold_out)($|_)|^[xy](te|test)$")
Y_NAME = re.compile(r"(?i)^(y|target|label|labels)(_.*)?$")
X_NAME = re.compile(r"(?i)^(x|features?)(_.*)?$")


def is_test_name(name: str) -> bool:
    return bool(_TEST_NAME.search(name))


def str_consts(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def lhs_name(assign: ast.Assign) -> str | None:
    if len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name):
        return assign.targets[0].id
    return None


def infer_label_refs(tree: ast.AST, known_targets: frozenset[str] = frozenset()) -> dict[str, frozenset[str]]:
    """Find `y = df["col"]`, `y = df.pop("col")`, `y = df.col == ...` and map df -> target columns."""
    refs: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        name = lhs_name(node)
        if not name or not Y_NAME.match(name):
            continue
        for sub in ast.walk(node.value):
            if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)
                    and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str)):
                refs[sub.value.id].add(sub.slice.value)
            elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "pop"
                  and isinstance(sub.func.value, ast.Name) and sub.args
                  and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str)):
                refs[sub.func.value.id].add(sub.args[0].value)
            elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.attr in known_targets:
                refs[sub.value.id].add(sub.attr)
    return {k: frozenset(v) for k, v in refs.items()}
