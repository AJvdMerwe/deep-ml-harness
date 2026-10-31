import json
import textwrap

import pytest

from ml_harness.domain.interfaces import ExperimentLogger
from ml_harness.domain.screening import GuardMode, ScreeningContext, ScreeningContextProvider, Severity
from ml_harness.evaluation.checks import default_checks
from ml_harness.evaluation.context import EdaSummaryContextProvider
from ml_harness.evaluation.middleware import LeakageGuardMiddleware
from ml_harness.evaluation.screener import LeakageScreener

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage


class StaticContext(ScreeningContextProvider):
    def __init__(self, **kw):
        self._ctx = ScreeningContext(**{k: frozenset(v) for k, v in kw.items()})

    def load(self):
        return self._ctx


def screen(code, **ctx):
    return LeakageScreener(default_checks(), StaticContext(**ctx)).screen(textwrap.dedent(code))


def rules(report, severity=None):
    return {f.rule_id for f in report.findings if severity is None or f.severity is severity}


CLEAN = """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier
    df = pd.read_parquet("/workspace/data/processed/churn_clean.parquet")
    y = df.pop("Churn")
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, stratify=y_tr)
    pre = OneHotEncoder().fit(X_tr)
    model = XGBClassifier()
    model.fit(pre.transform(X_tr), y_tr, eval_set=[(pre.transform(X_va), y_va)])
    print(model.predict(pre.transform(X_te)))
"""


def test_clean_script_passes():
    assert screen(CLEAN).findings == ()


def test_lk001_target_left_in_features():
    r = screen("""
        y = df["Churn"]
        X = df
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """)
    assert rules(r, Severity.BLOCK) == {"LK001"} and r.blocked


@pytest.mark.parametrize("build_x", [
    'X = df.drop(columns=["customerID"])',               # drops something else, target stays
    'X = df.select_dtypes("number")',                    # numeric 0/1 target survives
    'X = df[["tenure", "Churn"]]',                       # target explicitly selected
])
def test_lk001_variants_blocked(build_x):
    r = screen(f'y = df["Churn"]\n{build_x}\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n')
    assert "LK001" in rules(r, Severity.BLOCK)


@pytest.mark.parametrize("build_x", [
    'X = df.drop(columns=["Churn"])',
    'X = df.drop("Churn", axis=1)',
    'X = df[["tenure", "MonthlyCharges"]]',
    'df = df.drop(columns=["Churn"])\nX = df',
])
def test_lk001_safe_variants_pass(build_x):
    r = screen(f'y = df["Churn"]\n{build_x}\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n')
    assert "LK001" not in rules(r)


def test_lk001_uses_eda_context_for_attribute_access():
    r = screen('y = (df.Churn == "Yes")\nX = df\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n', target_columns=["Churn"])
    assert "LK001" in rules(r, Severity.BLOCK)


def test_lk002_scaler_fit_before_split():
    r = screen("""
        y = df.pop("Churn")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)
    """)
    assert "LK002" in rules(r, Severity.BLOCK)


def test_lk002_inline_constructor_and_fit_after_split_ok():
    bad = screen('y = df.pop("c")\nZ = SimpleImputer().fit_transform(df)\nA = train_test_split(Z, y)\n')
    ok = screen('y = df.pop("c")\nA, B, c, d = train_test_split(df, y)\nZ = SimpleImputer().fit_transform(A)\n')
    assert "LK002" in rules(bad) and "LK002" not in rules(ok)


def test_lk003_target_encoding_leak_and_safe_overwrite():
    leak = screen("""
        df["contract_churn_rate"] = df.groupby("Contract")["Churn"].transform("mean")
        y = df["Churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns=["Churn"]), y)
    """)
    assert "LK003" in rules(leak, Severity.BLOCK)
    safe = screen("""
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        y = df.pop("Churn")
        X_tr, X_te, y_tr, y_te = train_test_split(df, y)
    """)
    assert "LK003" not in rules(safe)


def test_lk003_assign_kwarg_form():
    r = screen('y = df["Churn"]\nd2 = df.assign(flag=lambda d: d["Churn"] * 2)\nA = train_test_split(d2, y)\n')
    assert "LK003" in rules(r)


@pytest.mark.parametrize("fit_line", [
    "model.fit(X_tr, y_tr, eval_set=[(X_test, y_test)])",
    "model.fit(X_te, y_te)",
    "scaler.fit_transform(X_test)",
])
def test_lk004_test_data_in_fit(fit_line):
    r = screen(f"y = df.pop('c')\nX_tr, X_test, y_tr, y_test = train_test_split(df, y)\n{fit_line}\n")
    assert "LK004" in rules(r, Severity.BLOCK)


def test_lk005_threshold_on_test_is_warning_only():
    r = screen("""
        y = df.pop("c")
        X_tr, X_test, y_tr, y_test = train_test_split(df, y)
        p, rc, thr = precision_recall_curve(y_test, proba)
        best = thr[np.argmax(p * rc)]
    """)
    assert rules(r) == {"LK005"} and not r.blocked


def test_lk006_suspect_columns_and_waiver():
    base = 'y = df.pop("Churn")\nX_tr, X_te, y_tr, y_te = train_test_split(df, y)\n'
    ctx = {"suspect_columns": ["cancel_date"]}
    assert "LK006" in rules(screen(base, **ctx), Severity.BLOCK)
    assert "LK006" not in rules(screen('df = df.drop(columns=["cancel_date"])\n' + base, **ctx))
    assert "LK006" not in rules(screen(base, suspect_columns=["cancel_date"], dropped_columns=["cancel_date"]))
    waived = screen("# leakage-waive: LK006 cancel_date is known at scoring time per data owner\n" + base, **ctx)
    assert not waived.blocked and [f.severity for f in waived.findings] == [Severity.INFO]


def test_non_waivable_rule_ignores_waiver():
    r = screen("""
        # leakage-waive: LK001 trust me, this is fine for sure
        y = df["Churn"]
        X = df
        A = train_test_split(X, y)
    """)
    assert r.blocked and "LK001" in rules(r, Severity.BLOCK)


def test_waiver_needs_a_real_reason():
    r = screen("# leakage-waive: LK006 ok\ny = df.pop('c')\nA = train_test_split(df, y)\n", suspect_columns=["leak"])
    assert r.blocked


def test_eda_scripts_and_syntax_errors_are_not_screened():
    assert screen('df["x"] = df["Churn"].astype(int)\nprint(df.corr())\n').findings == ()
    assert screen("def broken(:\n    model.fit(X_test)").findings == ()


def test_eda_summary_context_provider(tmp_path):
    f = tmp_path / "eda_summary.json"
    f.write_text(json.dumps({"target": "Churn", "leakage_suspects": [{"column": "cancel_date"}, "refund_flag"],
                             "dropped_columns": ["customerID"]}))
    ctx = EdaSummaryContextProvider(f).load()
    assert ctx.target_columns == {"Churn"} and ctx.suspect_columns == {"cancel_date", "refund_flag"}
    assert EdaSummaryContextProvider(tmp_path / "missing.json").load() == ScreeningContext()
    f.write_text("{not json")
    assert EdaSummaryContextProvider(f).load() == ScreeningContext()


# ---------------- middleware ----------------
class ListLogger(ExperimentLogger):
    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(dict(record))
        return "id"


LEAKY = 'y = df["Churn"]\nX = df\nA = train_test_split(X, y)\n'


def make_request(code, name="run_python"):
    return ToolCallRequest(tool_call={"name": name, "args": {"code": code}, "id": "call_1"},
                           tool=None, state={}, runtime=None)


def make_guard(mode=GuardMode.BLOCK):
    audit = ListLogger()
    return LeakageGuardMiddleware(LeakageScreener(default_checks(), StaticContext()), mode=mode, audit=audit), audit


def test_middleware_blocks_before_execution():
    guard, audit = make_guard()
    called = []
    result = guard.wrap_tool_call(make_request(LEAKY), lambda r: called.append(r))
    assert not called, "sandbox must not be reached"
    assert isinstance(result, ToolMessage) and result.status == "error" and result.tool_call_id == "call_1"
    assert "NOT executed" in result.content and "LK001" in result.content
    assert audit.records[0]["decision"] == "blocked" and len(audit.records[0]["script_sha256"]) == 16


def test_middleware_passes_clean_script_through():
    guard, audit = make_guard()
    ok = ToolMessage(content="exit_code=0", tool_call_id="call_1", name="run_python")
    assert guard.wrap_tool_call(make_request(textwrap.dedent(CLEAN)), lambda r: ok) is ok
    assert audit.records == []


def test_middleware_warn_mode_executes_and_annotates():
    guard, audit = make_guard(GuardMode.WARN)
    ok = ToolMessage(content="exit_code=0", tool_call_id="call_1", name="run_python")
    result = guard.wrap_tool_call(make_request(LEAKY), lambda r: ok)
    assert result.content.startswith("exit_code=0") and "LEAKAGE GUARD notes" in result.content
    assert audit.records[0]["decision"] == "passed_with_findings"


def test_middleware_ignores_other_tools():
    guard, _ = make_guard()
    sentinel = ToolMessage(content="listing", tool_call_id="call_1", name="ls")
    assert guard.wrap_tool_call(make_request(LEAKY, name="ls"), lambda r: sentinel) is sentinel


@pytest.mark.asyncio
async def test_middleware_async_path():
    pytest.importorskip("pytest_asyncio")
    guard, _ = make_guard()

    async def handler(_):
        raise AssertionError("must not execute")

    assert (await guard.awrap_tool_call(make_request(LEAKY), handler)).status == "error"
