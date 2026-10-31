from __future__ import annotations

import os
import subprocess
import time
import uuid

from ..config import SandboxPolicy
from ..domain.errors import SandboxUnavailableError
from ..domain.interfaces import CodeExecutor
from ..domain.models import ExecutionResult


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


class DockerExecutor(CodeExecutor):
    """Executes code in a throw-away, network-less, resource-capped container (code is passed on stdin)."""

    def __init__(self, policy: SandboxPolicy, docker_bin: str = "docker") -> None:
        self._policy = policy
        self._docker = docker_bin

    def ensure_ready(self) -> None:
        try:
            proc = subprocess.run(
                [self._docker, "image", "inspect", self._policy.image], capture_output=True, text=True, timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailableError(f"Docker is not available: {exc}") from exc
        if proc.returncode != 0:
            raise SandboxUnavailableError(
                f"Sandbox image '{self._policy.image}' not found. Build it with:\n"
                "  docker compose -f docker/docker-compose.yml --profile build build"
            )

    def build_command(self, container_name: str) -> list[str]:
        p = self._policy
        cmd = [
            self._docker, "run", "--rm", "-i", "--name", container_name,
            "--network", p.network,
            "--memory", p.memory, "--cpus", str(p.cpus), "--pids-limit", "256",
            "--read-only", "--tmpfs", "/tmp:rw,size=512m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp/mpl", "-e", "MPLBACKEND=Agg",
            "-w", "/workspace",
        ]
        if hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for m in p.mounts:
            cmd += ["-v", f"{m.host_path}:{m.container_path}" + (":ro" if m.read_only else "")]
        return cmd + [p.image, "python", "-u", "-"]

    def execute(self, code: str, *, timeout_s: int | None = None) -> ExecutionResult:
        timeout = timeout_s or self._policy.timeout_s
        name = f"mlh-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()
        try:
            proc = subprocess.run(
                self.build_command(name), input=code, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            self._kill(name)
            return ExecutionResult(
                exit_code=124, stdout=_decode(exc.stdout), stderr=_decode(exc.stderr),
                timed_out=True, duration_s=time.monotonic() - start,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(f"Docker binary not found: {exc}") from exc
        return ExecutionResult(
            exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            duration_s=time.monotonic() - start,
        )

    def _kill(self, name: str) -> None:
        subprocess.run([self._docker, "kill", name], capture_output=True, text=True, timeout=30, check=False)
