"""Abstractions (ports). High-level modules depend on these, never on concrete infrastructure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from deepagents import SubAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .models import ExecutionResult


class CodeExecutor(ABC):
    """Runs untrusted Python in isolation."""

    @abstractmethod
    def execute(self, code: str, *, timeout_s: int | None = None) -> ExecutionResult: ...


class ExperimentLogger(ABC):
    """Persists experiment records."""

    @abstractmethod
    def log(self, record: Mapping[str, Any]) -> str:
        """Persist a record and return its run id."""


class ToolProvider(ABC):
    """A cohesive group of agent tools."""

    @abstractmethod
    def tools(self) -> list[BaseTool]: ...


class SubAgentProvider(ABC):
    """Builds one DeepAgents sub-agent specification."""

    @abstractmethod
    def build(self) -> SubAgent: ...


class ChatModelFactory(ABC):
    """Creates the chat model the agents run on (hosted API or locally served)."""

    @abstractmethod
    def create(self) -> BaseChatModel | str: ...

    def health_check(self) -> None:
        """Raise ModelUnavailableError if the model cannot be used. Default: assume available."""

    @property
    def description(self) -> str:
        return type(self).__name__
