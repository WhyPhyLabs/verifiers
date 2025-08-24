from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import logging
import json
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    observation: str
    done: bool
    passed: bool
    info: dict[str, Any]


class TerminalBenchRunner(Protocol):
    """Protocol for running Terminal-Bench tasks."""

    def start(self, task_id: str) -> str:
        """Start a task and return the initial observation (e.g., shell prompt)."""

    def step(self, command: str) -> StepResult:
        """Execute a command and return the resulting observation and status."""


class StubRunner:
    """A minimal local runner for smoke tests and development."""

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


class HarnessRunner:
    """Terminal‑Bench runner using the official harness with the Terminus agent.

    This runner shells out to a user‑configurable module entry point for the
    Terminal‑Bench harness and runs a single task with the Terminus agent.

    Notes:
    - The Terminal‑Bench harness typically controls the agent end‑to‑end. Since
      TerminalBenchEnv follows a step(command) API, this runner executes the
      full harness on the first step and then returns the final outcome for any
      subsequent steps.
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str | None = None,
        entrypoint_module: str | None = None,
        entrypoint_args: list[str] | None = None,
        agent_name: str = "terminus",
        extra_env: dict[str, str] | None = None,
        timeout_sec: int = 1800,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.output_path = str(output_path or Path("./tb_runs").absolute())
        self.entrypoint_module = entrypoint_module or "terminal_bench.run"
        self.entrypoint_args = entrypoint_args or []
        self.agent_name = agent_name
        self.extra_env = extra_env or {}
        self.timeout_sec = timeout_sec

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self._current_task: str | None = None
        self._done: bool = False
        self._passed: bool = False
        self._observation: str = ""

    def start(self, task_id: str) -> str:
        self._current_task = task_id
        self._done = False
        self._passed = False
        self._observation = "Task prepared. Provide a shell command if desired; the harness will run Terminus for the task on first step."
        return self._observation

    def _run_harness(self) -> tuple[bool, str]:
        if not self._current_task:
            raise RuntimeError("HarnessRunner.start() must be called first.")

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg = {
                "dataset_path": self.dataset_path,
                "task_id": self._current_task,
                "output_path": self.output_path,
                "agent": self.agent_name,
            }
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            cmd = [
                sys.executable,
                "-m",
                self.entrypoint_module,
                "--config",
                str(cfg_path),
                *self.entrypoint_args,
            ]
            env = os.environ.copy()
            env.update(self.extra_env)
            try:
                logger.debug(
                    "TB HarnessRunner: launching module=%s task_id=%s output_path=%s agent=%s",
                    self.entrypoint_module,
                    self._current_task,
                    self.output_path,
                    self.agent_name,
                )
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=self.timeout_sec,
                )
                out = proc.stdout
                if proc.returncode != 0:
                    tail = out[-2000:] if out else ""
                    return False, f"TB harness exited with code {proc.returncode}.\n{tail}"
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or "") + (e.stderr or "")
                return False, f"TB harness timed out after {self.timeout_sec}s.\n{out[-2000:]}"

            # Attempt robust success detection across plausible report names
            resolved = False
            for name in ("final_report.json", "report.json", "results.json"):
                candidate = Path(self.output_path) / name
                if candidate.exists():
                    try:
                        data = json.loads(candidate.read_text())
                        if isinstance(data.get("success"), bool):
                            resolved = bool(data["success"])
                            break
                        if isinstance(data.get("resolved"), bool):
                            resolved = bool(data["resolved"])
                            break
                        rec = data.get("instances", {}).get(self._current_task)
                        if isinstance(rec, dict) and isinstance(rec.get("resolved"), bool):
                            resolved = bool(rec["resolved"])
                            break
                        if isinstance(data.get("resolved_instances"), list) and self._current_task in data.get("resolved_instances"):
                            resolved = True
                            break
                    except Exception:
                        pass
            return resolved, out

    def step(self, command: str) -> StepResult:  # noqa: ARG002
        if self._done:
            return StepResult(self._observation, True, self._passed, {"cached": True})

        resolved, logs = self._run_harness()
        self._done = True
        self._passed = resolved
        self._observation = ("All tests passed.\n" if resolved else "Tests failed.\n") + (logs[-2000:] if logs else "")
        return StepResult(self._observation, self._done, self._passed, {"task": self._current_task or "", "logs_tail": (logs[-2000:] if logs else "")})
