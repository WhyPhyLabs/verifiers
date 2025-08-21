from __future__ import annotations

from dataclasses import dataclass
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
