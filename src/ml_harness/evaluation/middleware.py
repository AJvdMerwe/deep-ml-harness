from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..domain.interfaces import ExperimentLogger
from ..domain.screening import GuardMode, ScreeningReport, ScriptScreener


class LeakageGuardMiddleware(AgentMiddleware):
    """Screens every `run_python` script for target leakage BEFORE it reaches the sandbox.

    - BLOCK mode: a script with blocking findings is not executed; the agent receives the findings as an
      error ToolMessage and is expected to fix and resubmit.
    - WARN mode: nothing is blocked; findings are appended to the tool output.
    Every non-empty report is written to the audit log (script hash, decision, findings).
    """

    def __init__(
        self,
        screener: ScriptScreener,
        *,
        mode: GuardMode = GuardMode.BLOCK,
        audit: ExperimentLogger | None = None,
        tool_name: str = "run_python",
    ) -> None:
        super().__init__()
        self._screener = screener
        self._mode = mode
        self._audit = audit
        self._tool_name = tool_name

    # -- shared logic -------------------------------------------------------------------------------------
    def _screen(self, request: ToolCallRequest) -> tuple[ScreeningReport | None, ToolMessage | None]:
        call = request.tool_call
        if call.get("name") != self._tool_name:
            return None, None
        code = (call.get("args") or {}).get("code", "")
        report = self._screener.screen(code if isinstance(code, str) else "")
        blocked = report.blocked and self._mode is GuardMode.BLOCK
        if self._audit and report.findings:
            self._audit.log({
                "event": "leakage_screen",
                "mode": self._mode.value,
                "decision": "blocked" if blocked else ("passed_with_findings"),
                "script_sha256": hashlib.sha256(code.encode()).hexdigest()[:16],
                "findings": [{"rule": f.rule_id, "severity": f.severity.value, "line": f.line, "message": f.message}
                             for f in report.findings],
            })
        if blocked:
            return report, ToolMessage(content=report.render_block(), tool_call_id=call["id"],
                                       name=self._tool_name, status="error")
        return report, None

    @staticmethod
    def _annotate(result: Any, report: ScreeningReport | None) -> Any:
        if report and report.findings and isinstance(result, ToolMessage) and isinstance(result.content, str):
            return result.model_copy(update={"content": f"{result.content}\n\n{report.render_notes()}"})
        return result

    # -- AgentMiddleware hooks ----------------------------------------------------------------------------
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        report, rejection = self._screen(request)
        if rejection is not None:
            return rejection
        return self._annotate(handler(request), report)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        report, rejection = self._screen(request)
        if rejection is not None:
            return rejection
        return self._annotate(await handler(request), report)
