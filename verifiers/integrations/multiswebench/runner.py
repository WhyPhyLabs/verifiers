from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


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


class HarnessRunner:
    """Multi‑SWE‑Bench runner backed by the official harness.

    This runner shells out to `python -m multi_swe_bench.harness.run_evaluation`
    for a single instance at a time. It expects Docker to be available.

    task_id format: "<org>__<repo>-<number>" (e.g., "axios__axios-5919").

    Args:
        dataset_file: Path to a dataset JSONL file (one of the downloaded
            Multi‑SWE‑Bench dataset shards).
        workdir: Working directory for the harness (build caches, etc.).
        output_dir: Output directory for logs and reports.
        repo_dir: Directory where target repositories can be cloned.
        max_workers: Parallelism for the harness (kept low for single instance).
        force_build: Whether to force rebuild images.
        cache_level: Image cache level ("env" recommended).
        log_level: Harness log level.
        extra_env: Extra env vars to set for subprocess calls.
    """

    def __init__(
        self,
        dataset_file: str,
        workdir: str | None = None,
        output_dir: str | None = None,
        repo_dir: str | None = None,
        max_workers: int = 1,
        force_build: bool = False,
        cache_level: str = "env",
        log_level: str = "INFO",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.dataset_file = str(dataset_file)
        self.workdir = str(workdir or Path("./msb_work").absolute())
        self.output_dir = str(output_dir or Path("./msb_out").absolute())
        self.repo_dir = str(repo_dir or Path("./msb_repos").absolute())
        self.max_workers = max_workers
        self.force_build = force_build
        self.cache_level = cache_level
        self.log_level = log_level
        self.extra_env = extra_env or {}

        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.repo_dir).mkdir(parents=True, exist_ok=True)

        # Preload dataset index for quick lookup
        self._index: dict[str, dict[str, Any]] = {}
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                inst_id = f"{item['org']}__{item['repo']}-{item['number']}"
                self._index[inst_id] = item

        self._current_task: str | None = None

    def _parse_task(self, task_id: str) -> dict[str, Any]:
        if task_id not in self._index:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Ensure it exists in {self.dataset_file}"
            )
        return self._index[task_id]

    def _run_harness(self, config: dict[str, Any]) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps(config), encoding="utf-8")
            cmd = [
                shutil.which("python") or "python",
                "-m",
                "multi_swe_bench.harness.run_evaluation",
                "--config",
                str(cfg_path),
            ]
            env = os.environ.copy()
            env.update(self.extra_env)
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            out = proc.stdout
            # Best-effort parse: when single instance is run, a per-instance
            # report.json is created under logs dir; but also a final_report.json
            # may be created under output_dir.
            # We scan output_dir for any report.json to read a resolved flag.
            reports = list(Path(self.output_dir).rglob("report.json"))
            resolved = False
            if reports:
                try:
                    data = json.loads(reports[0].read_text())
                    # report keyed by instance id
                    for v in data.values():
                        if isinstance(v, dict) and "resolved" in v:
                            resolved = bool(v["resolved"])
                            break
                except Exception:
                    pass
            return resolved, out

    def start(self, task_id: str) -> str:
        self._current_task = task_id
        _ = self._parse_task(task_id)
        return (
            "Multi‑SWE‑Bench instance prepared. Provide a unified diff patch to apply.\n"
            "Format: output of `git diff -U` with proper file paths."
        )

    def apply_patch(self, patch_text: str) -> PatchStepResult:
        if not self._current_task:
            raise RuntimeError("HarnessRunner.start() must be called first.")
        item = self._parse_task(self._current_task)
        inst_id = self._current_task

        # Write a single‑instance patch file in JSONL format
        patch_dir = Path(self.workdir) / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{inst_id}.jsonl"
        patch_obj = {
            "org": item["org"],
            "repo": item["repo"],
            "number": item["number"],
            "fix_patch": patch_text,
        }
        patch_path.write_text(json.dumps(patch_obj) + "\n", encoding="utf-8")

        # Build harness config for a single instance
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "mode": "evaluation",
            "workdir": self.workdir,
            "patch_files": [str(patch_path)],
            "dataset_files": [self.dataset_file],
            "force_build": self.force_build,
            "output_dir": self.output_dir,
            "specifics": [inst_id],
            "skips": [],
            "repo_dir": self.repo_dir,
            "need_clone": True,
            "global_env": [],
            "clear_env": True,
            "stop_on_error": True,
            "max_workers": self.max_workers,
            "max_workers_build_image": max(1, self.max_workers),
            "max_workers_run_instance": max(1, self.max_workers),
            "log_dir": str(log_dir),
            "log_level": self.log_level,
        }

        resolved, logs = self._run_harness(cfg)
        obs = "All tests passed.\n" if resolved else "Tests still failing.\n"
        info = {"instance": inst_id, "logs_tail": logs[-2000:]}
        done = resolved  # stop when resolved, otherwise allow more tries
        return PatchStepResult(observation=obs, done=done, passed=resolved, info=info)
