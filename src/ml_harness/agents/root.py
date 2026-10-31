from __future__ import annotations

from typing import Any, Sequence

from deepagents import create_deep_agent

from ..domain.interfaces import SubAgentProvider, ToolProvider
from . import prompts


class ResearchAgentBuilder:
    """Assembles the root agent from injected parts; knows nothing about Docker, files or concrete sub-agents."""

    def __init__(
        self,
        *,
        model: Any,  # 'provider:model' string or a BaseChatModel instance
        backend: Any,
        tool_providers: Sequence[ToolProvider],
        subagent_providers: Sequence[SubAgentProvider],
        skill_sources: Sequence[str] = ("/skills/shared/",),
        system_prompt: str = prompts.ROOT_PROMPT,
        checkpointer: Any | None = None,
        recursion_limit: int = 150,
    ) -> None:
        self._model = model
        self._backend = backend
        self._tool_providers = tuple(tool_providers)
        self._subagent_providers = tuple(subagent_providers)
        self._skill_sources = list(skill_sources)
        self._system_prompt = system_prompt
        self._checkpointer = checkpointer
        self._recursion_limit = recursion_limit

    def build(self):
        agent = create_deep_agent(
            model=self._model,
            tools=[t for p in self._tool_providers for t in p.tools()],
            system_prompt=self._system_prompt,
            subagents=[p.build() for p in self._subagent_providers],
            skills=self._skill_sources,
            backend=self._backend,
            checkpointer=self._checkpointer,
        )
        return agent.with_config({"recursion_limit": self._recursion_limit})
