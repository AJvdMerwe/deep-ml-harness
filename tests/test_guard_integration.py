"""Runs the guard inside a real LangChain agent graph with a scripted model (no API key, no Docker)."""
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ml_harness.domain.screening import GuardMode
from ml_harness.evaluation.checks import default_checks
from ml_harness.evaluation.middleware import LeakageGuardMiddleware
from ml_harness.evaluation.screener import LeakageScreener
from ml_harness.tools.code_execution import CodeExecutionTools

from test_leakage_guard import CLEAN, LEAKY, StaticContext
import textwrap


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def run_agent(scripts, executor, mode=GuardMode.BLOCK):
    msgs = [AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": c}, "id": f"c{i}"}])
            for i, c in enumerate(scripts)] + [AIMessage(content="done")]
    guard = LeakageGuardMiddleware(LeakageScreener(default_checks(), StaticContext()), mode=mode)
    agent = create_agent(ScriptedModel(messages=iter(msgs)), tools=CodeExecutionTools(executor).tools(),
                         middleware=[guard])
    return agent.invoke({"messages": [HumanMessage("train")]})


def test_blocked_script_never_reaches_executor_and_agent_can_recover(fake_executor):
    result = run_agent([LEAKY, textwrap.dedent(CLEAN)], fake_executor)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert "NOT executed" in tool_msgs[0].content and tool_msgs[0].status == "error"
    assert "fake-ok" in tool_msgs[1].content
    assert fake_executor.calls == [textwrap.dedent(CLEAN)]  # only the fixed script ran


def test_warn_mode_runs_leaky_script_with_notes(fake_executor):
    result = run_agent([LEAKY], fake_executor, GuardMode.WARN)
    tool_msg = next(m for m in result["messages"] if m.type == "tool")
    assert fake_executor.calls == [LEAKY] and "LEAKAGE GUARD notes" in tool_msg.content
