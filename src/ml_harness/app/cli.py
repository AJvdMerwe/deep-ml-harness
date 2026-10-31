from __future__ import annotations

import argparse
import sys
import uuid

from ..config import Settings
from ..domain.errors import ModelUnavailableError, SandboxUnavailableError
from ..infrastructure.docker_executor import DockerExecutor
from ..infrastructure.model_factories import create_model_factory
from .composition import build_agent


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(prog="ml-harness", description="Deep-agent data research & ML harness")
    parser.add_argument("query", nargs="?", help='e.g. "Train a XGBoost model on our churn data"')
    parser.add_argument("--thread", default=None, help="thread id (reuse to continue a session)")
    parser.add_argument("--stream", action="store_true", help="print each agent step as it happens")
    parser.add_argument("--check", action="store_true", help="verify Docker + sandbox image, then exit")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    settings.paths.ensure()
    executor = DockerExecutor(settings.sandbox)
    try:
        executor.ensure_ready()
    except SandboxUnavailableError as exc:
        print(f"[sandbox] {exc}", file=sys.stderr)
        return 2
    factory = create_model_factory(settings.model)
    try:
        factory.health_check()
    except ModelUnavailableError as exc:
        print(f"[model] {exc}", file=sys.stderr)
        return 3
    if args.check:
        print(f"OK: sandbox image '{settings.sandbox.image}' ready; model: {factory.description}; "
              f"project root {settings.paths.root}")
        return 0
    if not args.query:
        parser.error("query is required (or use --check)")

    agent = build_agent(settings, executor, model=factory.create())
    inputs = {"messages": [{"role": "user", "content": args.query}]}
    config = {"configurable": {"thread_id": args.thread or uuid.uuid4().hex}}

    if args.stream:
        seen = 0
        for chunk in agent.stream(inputs, config=config, stream_mode="values"):
            messages = chunk.get("messages", [])
            for message in messages[seen:]:
                message.pretty_print()
            seen = len(messages)
    else:
        result = agent.invoke(inputs, config=config)
        print(_text(result["messages"][-1].content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
