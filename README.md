# ml-harness: a Deep Agents harness for data research and machine learning

```
[ USER QUERY: "Train a XGBoost model on our churn data" ]
                           │
                           ▼
        ┌────────────────────────────────────┐
        │ ROOT DEEP RESEARCH AGENT           │  write_todos plan · list_datasets · final report
        │ (agents/root.py + ROOT_PROMPT)     │  skills: /skills/shared/
        └───────────────┬────────────────────┘
          task tool     │
     ┌──────────────────┴───────────────────┐
     ▼                                      ▼
┌───────────────────────┐        ┌───────────────────────┐
│ SUB-AGENT: data-eda   │        │ SUB-AGENT: ml-training│
│ run_python · profile  │        │ run_python · log_exp  │
│ skills: shared + eda  │        │ skills: shared+training│
└──────────┬────────────┘        └──────────┬────────────┘
           └──────────────┬─────────────────┘
                          │ CodeExecutor port
                          ▼
   DockerExecutor → throw-away container (no network, read-only rootfs, caps dropped, mem/cpu capped)
   ┌─────────────────────────────────────────────────────────────┐
   │ data/raw (ro) · data/processed │ artifacts/{models,reports,logs} │ plots/ │
   └─────────────────────────────────────────────────────────────┘
```

## Using a locally hosted LLM

The model is behind a `ChatModelFactory` port (`infrastructure/model_factories.py`); the agents never know which provider runs them. `ml-harness --check` verifies the server is reachable **and** the model is available.

| Server | Install | `.env` |
|---|---|---|
| Ollama | `pip install -e ".[ollama]"`, `ollama pull qwen3:14b` | `ML_HARNESS_MODEL=ollama:qwen3:14b` |
| vLLM | `pip install -e ".[openai]"` | `ML_HARNESS_MODEL_PROVIDER=vllm`, `ML_HARNESS_MODEL=<served model id>`, `ML_HARNESS_MODEL_BASE_URL=http://localhost:8000/v1` |
| LM Studio | `.[openai]` | provider `lmstudio`, base URL `http://localhost:1234/v1` |
| llama.cpp server | `.[openai]` | provider `llamacpp`, base URL `http://localhost:8080/v1` |
| Hosted (Anthropic etc.) | `.[anthropic]` | `ML_HARNESS_MODEL=anthropic:claude-sonnet-5-5` + API key |

Requirements and tips for local models:
- **Tool calling is mandatory** (DeepAgents is tool-driven). Use a tool-capable model (e.g. Qwen3/Qwen2.5-Instruct, Llama 3.1/3.3, Mistral-Nemo, gpt-oss). vLLM needs `--enable-auto-tool-choice --tool-call-parser <parser>` (e.g. `hermes` for Qwen); llama.cpp needs `--jinja`.
- **Context window:** the agent prompt plus skills plus tool output needs roughly 16-32K tokens. Ollama silently defaults to a few thousand, so the harness sets `num_ctx=32768` (`ML_HARNESS_MODEL_NUM_CTX`). Raise `--max-model-len` / `-c` on other servers.
- **Model size:** about 14B+ (ideally 30B+) is realistic for multi-step planning and XGBoost scripting; small models loop or emit broken tool calls. Keep temperature at 0.
- Local generation is slower; `ML_HARNESS_MODEL_TIMEOUT_S` and the recursion limit are tunable. The leakage guard matters more with weaker models.
- Only the LLM is local-or-remote; code still runs in the network-less Docker sandbox, so the data never leaves your machine when the model is local.
- Add another provider by implementing `ChatModelFactory` and adding it to `_REGISTRY`.

## Evaluation guard: leakage screening before execution

`evaluation/` adds a middleware layer on the **ml-training** sub-agent. Every `run_python` call is statically analysed
(AST, no data access, milliseconds) before the sandbox is launched:

```
ml-training LLM --run_python(code)--> LeakageGuardMiddleware --screen--> LeakageScreener --> [LK001..LK006 checks]
                                          |  blocked: ToolMessage(error) with findings, sandbox NOT launched
                                          '--> passes: DockerExecutor runs it (findings appended as notes)
```

| Rule | What it catches |
|---|---|
| LK001 | label column left in the feature matrix |
| LK002 | preprocessing/feature selection fit before `train_test_split` |
| LK003 | features derived from the label (full-frame target encoding), waivable |
| LK004 | test data used in `fit` / `eval_set` / early stopping |
| LK005 | threshold or hyperparameters chosen with test labels (warning), waivable |
| LK006 | columns the EDA stage flagged in `leakage_suspects` but training never drops, waivable |

- **Context-aware:** `EdaSummaryContextProvider` reads `target`, `leakage_suspects`, `dropped_columns` from `artifacts/reports/eda_summary.json`, so EDA findings are enforced at training time. Targets are also inferred from the script (`y = df.pop(...)`).
- **Modes:** `ML_HARNESS_LEAKAGE_MODE=block` (default) | `warn` (never blocks, annotates output) | `off` (not attached).
- **Waivers:** `# leakage-waive: LK006 <reason>` on waivable rules only; LK001, LK002 and LK004 cannot be waived. Waived findings stay visible.
- **Audit:** every report with findings is appended to `artifacts/logs/leakage_audit.jsonl` (script hash, decision, findings).
- **Extending:** add a rule by subclassing `ScriptCheck` and adding it to `default_checks()`. Swap the context source by implementing `ScreeningContextProvider`.
- **Limits:** this is heuristic static analysis (it follows naming conventions such as `y_*`, `X_*`, `*_test`). It catches the common, mechanical leaks; it cannot prove absence of leakage. Data-level checks (single-feature AUC, duplicate rows across splits) belong in an additional `ScriptCheck` backed by a sandboxed probe, or in the EDA stage.

## Layout

| Directory | Purpose |
|---|---|
| `data/` | `raw/` (mounted **read-only** in the sandbox) and `processed/` |
| `artifacts/` | `models/` (.pkl/.onnx), `reports/` (JSON/MD), `logs/` (`experiments.jsonl`) |
| `plots/` | PNGs, `eda/` and `training/` |
| `src/ml_harness/` | all source code (layers below) |
| `docker/` | `Dockerfile.sandbox`, `requirements-sandbox.txt`, `docker-compose.yml` |
| `environment.yml` | conda environment for the host (agent + CLI) |
| `skills/` | agentskills-format `SKILL.md` folders: `shared/`, `eda/`, `training/` |
| `tests/` | unit tests (no Docker or API key needed) |

## Code layers and SOLID

```
domain/          ports + value objects (CodeExecutor, ExperimentLogger, ToolProvider, SubAgentProvider, ScriptCheck, ScriptScreener, Finding)
evaluation/      leakage rules (checks.py), screener, EDA context provider, LeakageGuardMiddleware
infrastructure/  adapters (DockerExecutor, JsonlExperimentLogger, WorkspaceBackendBuilder)
tools/           ToolProviders: CodeExecutionTools, DataCatalogTools, ExperimentLogTools
agents/          prompts.py (text only), subagents.py (providers), root.py (ResearchAgentBuilder)
app/             composition.py (the only place concrete classes are chosen), cli.py
config.py        environment -> immutable Settings; path policy
```

| Principle | Where |
|---|---|
| **S**ingle responsibility | One reason to change per class: executing code, logging runs, exposing files, building one sub-agent, assembling the root agent. Prompts live apart from logic. |
| **O**pen/closed | Add a sub-agent (e.g. `ForecastingSubAgentProvider`) or tool group by adding a class and one line in `composition.py`; `ResearchAgentBuilder` never changes. |
| **L**iskov | Any `CodeExecutor` (Docker, remote sandbox, fake in tests) is substitutable; subclasses of `BaseSubAgentProvider` only fill in declared hooks. |
| **I**nterface segregation | Four tiny ports; a tool needing only `ExperimentLogger` is not forced to know about execution. |
| **D**ependency inversion | Tools/agents depend on `domain.interfaces`; `composition.py` injects `DockerExecutor`, the logger and the backend. Tests swap in `FakeExecutor`. |

## Quick start

```bash
conda env create -f environment.yml        # Python 3.12 + the harness (extras: dev, ollama)
conda activate ml-harness
cp .env.example .env                       # choose a model block (local Ollama is the default)
docker compose -f docker/docker-compose.yml --profile build build   # sandbox image
ml-harness --check
ml-harness-sample-data                     # synthetic data/raw/churn.csv (or drop in your own)
ml-harness --stream "Train a XGBoost model on our churn data"
pytest
```

Run `conda env create` from the project root (the folder containing `environment.yml` and `pyproject.toml`).

Conda notes:
- **Different LLM backend:** edit the extras in `environment.yml` (`openai` for vLLM / LM Studio / llama.cpp, `anthropic` for hosted) and run `conda env update -f environment.yml --prune`, or just `python -m pip install -e ".[openai]"` inside the env.
- **Command not found:** the `ml-harness` command lives in the active env. Run `conda activate ml-harness` and `hash -r`. Use `python -m pip` rather than bare `pip` so installs land in the env. Fallback: `python -m ml_harness.app.cli --check`.
- **Reset:** `conda env remove -n ml-harness`, then recreate.
- The sandbox image is built with pip inside Docker (small and reproducible), independent of conda.

Outputs appear under `data/processed/`, `artifacts/`, `plots/`.

## Security model
- Agent-written code runs **only** in the container: `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, pids/memory/cpu limits, timeout with `docker kill`, host uid mapped.
- Raw data is mounted read-only in the sandbox. The agent's *file tools* are routed only to `data/`, `artifacts/`, `plots/`, `skills/`; `src/` and `docker/` are unreachable. Note the file tools can still write under `data/raw/` on the host (the prompt forbids it); mount-level protection applies to code execution only. Wrap `FilesystemBackend` if you need hard enforcement.
- The sandbox has no credentials. The API key stays on the host.
- Docker socket access = host-level privilege. Run the harness on a trusted machine or swap `DockerExecutor` for a remote sandbox (implement `CodeExecutor`).

## Extending
1. New skill: add `skills/<group>/<name>/SKILL.md` and list the group in a provider's `skill_sources`.
2. New sub-agent: subclass `BaseSubAgentProvider`, register in `composition.py`.
3. New sandbox: implement `CodeExecutor`, pass it to `build_agent(settings, executor)`.
4. Persistent sessions: replace `MemorySaver` in `composition.py` with a SQLite/Postgres checkpointer.
