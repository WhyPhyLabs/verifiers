from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class PatchStepResult:
    observation: str
    done: bool
    passed: bool
    info: dict[str, Any]


class MultiSWERunner(Protocol):
    """Protocol for running Multi‑SWE‑Bench tasks.

    A concrete implementation should:
    - fetch/prepare the target repo for a specific task id
    - accept patches (e.g., unified diffs) proposed by the agent
    - apply the patch, run the benchmark's tests, and report pass/fail
    """

    def start(self, task_id: str) -> str:
        """Prepare task and return initial observation (e.g., failing tests)."""

    def apply_patch(self, patch_text: str) -> PatchStepResult:
        """Apply a patch and return test results and status."""


class StubRunner:
    """A minimal runner for smoke tests and development.

    Marks success when a proposed patch contains an expected token.
    """

    def __init__(self, expected_token: str = "PASS_FIX", max_attempts: int = 3):
        self.expected_token = expected_token
        self.max_attempts = max_attempts
        self.attempts = 0
        self.started = False

    def start(self, task_id: str) -> str:  # noqa: ARG002
        self.attempts = 0
        self.started = True
        return "Failing tests: 1\n- test_example::test_should_pass (currently failing)\nSuggest a patch (unified diff) to fix."

    def apply_patch(self, patch_text: str) -> PatchStepResult:
        if not self.started:
            raise RuntimeError("StubRunner.start() must be called before apply_patch().")
        self.attempts += 1
        done = False
        passed = False
        info: dict[str, Any] = {"attempts": self.attempts}
        if self.expected_token in patch_text:
            done = True
            passed = True
            obs = "All tests passed.\n"
        else:
            obs = "Tests still failing.\n"
            if self.attempts >= self.max_attempts:
                done = True
                passed = False
        return PatchStepResult(observation=obs, done=done, passed=passed, info=info)
