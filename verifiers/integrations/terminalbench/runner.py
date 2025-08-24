from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from subprocess import run
from typing import Any, Protocol


@dataclass
class StepResult:
    observation: str
    done: bool
    passed: bool
    info: dict[str, Any]


class TerminalBenchRunner(Protocol):
    """Protocol for running Terminal-Bench tasks.

    A concrete implementation should drive a Terminal-Bench task lifecycle,
    returning observations after executing agent-proposed commands, and report
    completion/pass status. Implementations may talk to the official harness
    via MCP, direct Python API, or CLI/REST wrappers.
    """

    def start(self, task_id: str) -> str:
        """Start a task and return the initial observation (e.g., shell prompt)."""

    def step(self, command: str) -> StepResult:
        """Execute a command and return the resulting observation and status."""


class StubRunner:
    """A minimal local runner for smoke tests and development.

    Simulates a simple Terminal-Bench-like task that expects a specific command
    sequence. Useful for CI and development environments without the harness.
    """

    def __init__(self, expected_command: str = "echo hello", max_steps: int = 5):
        self.expected_command = expected_command
        self.max_steps = max_steps
        self.steps = 0
        self.started = False

    def start(self, task_id: str) -> str:  # noqa: ARG002
        self.steps = 0
        self.started = True
        return "root@stub:/app# "

    def step(self, command: str) -> StepResult:
        if not self.started:
            raise RuntimeError("StubRunner.start() must be called before step().")
        self.steps += 1
        done = False
        passed = False
        info: dict[str, Any] = {"steps": self.steps}
        if command.strip() == self.expected_command:
            done = True
            passed = True
            obs = "OK\n"
        else:
            obs = f"/bin/sh: 1: {command.strip()}: not found\n"
            if self.steps >= self.max_steps:
                done = True
                passed = False
        return StepResult(observation=obs, done=done, passed=passed, info=info)

logger = logging.getLogger(__name__)


class HarnessRunner:
    """Harness-backed runner using a Terminal-Bench entrypoint module.

    This class shells out to a provided entrypoint module (default:
    `terminal_bench.run`) that is expected to manage the Terminal‑Bench task
    lifecycle. It executes a single command per step by invoking the module
    with the task id, agent name, dataset path, and the command to execute.

    Notes:
    - This is a pragmatic adapter around the harness' CLI interface. It relies
      on the entrypoint supporting a "single-step" invocation contract. If the
      upstream interface differs, adjust `entrypoint_module` and `extra_args` to
      match. Errors and return codes are surfaced in `info`.
    - Timeouts and return-code checks are included for robustness.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        output_path: str = "./tb_runs",
        entrypoint_module: str = "terminal_bench.run",
        agent_name: str = "terminus",
        timeout_sec: int = 1800,
        extra_args: list[str] | None = None,
    ) -> None:
        # Preflight: dataset path must exist
        import os as _os

        if not _os.path.isdir(dataset_path):
            raise ValueError(f"dataset_path does not exist or is not a directory: {dataset_path}")
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.entrypoint_module = entrypoint_module
        self.agent_name = agent_name
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []
        self._started = False

    def start(self, task_id: str) -> str:
        self._task_id = task_id
        self._started = True
        return f"{task_id}> "

    def step(self, command: str) -> StepResult:
        if not self._started:
            raise RuntimeError("HarnessRunner.start() must be called before step().")
        # Invoke the entrypoint in a single-step mode. The concrete CLI contract is
        # entrypoint-dependent; we pass conservative, self-describing flags.
        args = [
            "--dataset",
            self.dataset_path,
            "--task",
            self._task_id,
            "--agent",
            self.agent_name,
            "--command",
            command,
            "--output",
            self.output_path,
            *self.extra_args,
        ]
        logger.debug(
            "TB HarnessRunner: task_id=%s agent=%s output_path=%s command=%s",
            getattr(self, "_task_id", None),
            self.agent_name,
            self.output_path,
            command,
        )
        proc = run(
            [sys.executable, "-m", self.entrypoint_module, *args],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        obs = proc.stdout or ""
        info: dict[str, Any] = {
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
        # Persist full logs for debugging
        try:
            import os as _os
            _os.makedirs(self.output_path, exist_ok=True)
            with open(f"{self.output_path}/last_stdout.txt", "w", encoding="utf-8") as f:
                f.write(proc.stdout or "")
            with open(f"{self.output_path}/last_stderr.txt", "w", encoding="utf-8") as f:
                f.write(proc.stderr or "")
        except Exception:
            pass
        # Consider success when the process returns 0 and prints a success marker.
        passed = proc.returncode == 0 and ("SUCCESS" in obs or "OK" in obs)
        done = passed or proc.returncode != 0
        return StepResult(observation=obs, done=done, passed=passed, info=info)
