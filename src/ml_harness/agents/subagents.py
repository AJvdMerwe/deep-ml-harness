from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

from deepagents import SubAgent

from ..domain.interfaces import SubAgentProvider, ToolProvider
from . import prompts


class BaseSubAgentProvider(SubAgentProvider):
    """Template method: subclasses declare identity, prompt and skills; tool wiring is injected."""

    def __init__(
        self,
        tool_providers: Sequence[ToolProvider],
        model: str | None = None,
        middleware: Sequence[Any] = (),
    ) -> None:
        self._tool_providers = tuple(tool_providers)
        self._model = model
        self._middleware = tuple(middleware)

    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def description(self) -> str: ...
    @property
    @abstractmethod
    def system_prompt(self) -> str: ...
    @property
    @abstractmethod
    def skill_sources(self) -> list[str]: ...

    def build(self) -> SubAgent:
        spec = SubAgent(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            tools=[t for p in self._tool_providers for t in p.tools()],
            skills=self.skill_sources,
        )
        if self._model:
            spec["model"] = self._model
        if self._middleware:
            spec["middleware"] = list(self._middleware)
        return spec


class EdaSubAgentProvider(BaseSubAgentProvider):
    name = "data-eda"
    description = (
        "Explores and cleans a dataset: profiling, missing values, type fixes, leakage checks, distributions, "
        "correlations. Saves cleaned parquet, plots and an EDA summary JSON. Use BEFORE any modelling."
    )
    system_prompt = prompts.EDA_PROMPT
    skill_sources = ["/skills/shared/", "/skills/eda/"]


class TrainingSubAgentProvider(BaseSubAgentProvider):
    name = "ml-training"
    description = (
        "Trains and evaluates XGBoost models on a CLEANED dataset: stratified split, hyperparameter search loop, "
        "threshold tuning, metrics, plots, model export (.pkl/.onnx) and experiment logging."
    )
    system_prompt = prompts.TRAINING_PROMPT
    skill_sources = ["/skills/shared/", "/skills/training/"]
