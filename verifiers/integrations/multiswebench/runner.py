from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any, Protocol


@dataclass
class EvalResult:
    """Outcome of evaluating a single instance/patch."""

    resolved: bool
    info: dict[str, Any]


class MultiSWERunner(Protocol):
    """Protocol for evaluating a single Multi-SWE-Bench instance + patch."""

    def evaluate(
        self,
        *,
        instance_id: str,
        patch: str,
    ) -> EvalResult:  # pragma: no cover - protocol
        ...


class StubRunner:
    """A minimal runner for CI that treats a magic token as success.

    If the provided patch contains the token "FIX_PATCH", we mark it as resolved,
    otherwise unresolved. This avoids external dependencies while preserving the
    control flow of the environment.
    """

    def __init__(self, success_token: str = "FIX_PATCH") -> None:
        self.success_token = success_token

    def evaluate(self, *, instance_id: str, patch: str) -> EvalResult:  # noqa: ARG002
        passed = self.success_token in patch
        return EvalResult(resolved=passed, info={"stub": True})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tail(text: str, n: int = 2000) -> str:
    return text[-n:]


def _run_entrypoint(
    *,
    module: str,
    args: list[str],
    cwd: str | None,
    timeout_sec: int,
) -> CompletedProcess:
    cmd = [sys.executable, "-m", module, *args]
    return run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout_sec)


def _report_indicates_resolved(report_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Heuristically determine success from Multi-SWE reports.

    Prefer final_report.json (single-source-of-truth in official harness). Fall back
    to report.json and try common shapes.
    """
    info: dict[str, Any] = {}
    final = report_dir / "final_report.json"
    if final.exists():
        try:
            data = json.loads(final.read_text(encoding="utf-8"))
            info["final_report"] = True
            # Common shapes: {"resolved_instances":["<id>", ...]} or
            # {"instances": {"<id>": {"resolved": true}}}
            if isinstance(data, dict):
                if isinstance(data.get("resolved_instances"), list):
                    resolved = bool(data.get("resolved_instances"))
                    info["resolved_instances"] = data["resolved_instances"]
                    return resolved, info
                inst = data.get("instances")
                if isinstance(inst, dict):
                    # consider resolved if any instance resolved
                    any_resolved = any(
                        isinstance(v, dict) and v.get("resolved") for v in inst.values()
                    )
                    return bool(any_resolved), info
        except Exception:
            pass

    legacy = report_dir / "report.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            info["legacy_report"] = True
            if isinstance(data, dict):
                if isinstance(data.get("resolved_instances"), list):
                    resolved = bool(data.get("resolved_instances"))
                    info["resolved_instances"] = data["resolved_instances"]
                    return resolved, info
                inst = data.get("instances")
                if isinstance(inst, dict):
                    any_resolved = any(
                        isinstance(v, dict) and v.get("resolved") for v in inst.values()
                    )
                    return bool(any_resolved), info
        except Exception:
            pass
    return False, info


class HarnessRunner:
    """Runner that shells out to the official Multi-SWE-Bench harness.

    This runner writes a single-instance config and patch file into a temporary
    directory and invokes the harness module with that config. It then parses
    the produced report to determine success.

    The runner is intentionally entrypoint-parameterized to avoid hard deps and
    allow evolving upstream interfaces.
    """

    def __init__(
        self,
        *,
        dataset_file: str,
        output_dir: str = "./msb_runs",
        entrypoint_module: str = "multi_swe_bench.harness.run_evaluation",
        timeout_sec: int = 1800,
        extra_args: list[str] | None = None,
    ) -> None:
        # Preflight
        if not Path(dataset_file).is_file():
            raise ValueError(f"dataset_file does not exist or is not a file: {dataset_file}")
        self.dataset_file = dataset_file
        self.output_dir = Path(output_dir)
        self.entrypoint_module = entrypoint_module
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []

    def evaluate(self, *, instance_id: str, patch: str) -> EvalResult:
        # Layout temp working dir
        with tempfile.TemporaryDirectory(prefix="msb_") as tmp:
            tmpdir = Path(tmp)
            work_out = tmpdir / "out"
            work_out.mkdir(parents=True, exist_ok=True)

            # Write a single-instance patch jsonl file as expected by harness
            patches_path = tmpdir / "patches.jsonl"
            patch_item = {
                "instance_id": instance_id,
                "patch": patch,
            }
            _write_text(patches_path, json.dumps(patch_item) + "\n")

            # Build a minimal config dict. Upstream may accept additional keys; we pass
            # only conservative, commonly supported fields.
            cfg = {
                "dataset_file": os.fspath(Path(self.dataset_file).resolve()),
                "patches_path": os.fspath(patches_path),
                "output_dir": os.fspath(work_out),
                "max_workers": 1,
                "instances": [instance_id],
            }
            cfg_path = tmpdir / "config.json"
            _write_text(cfg_path, json.dumps(cfg))

            # Invoke harness
            args = ["--config", os.fspath(cfg_path), *self.extra_args]
            proc = _run_entrypoint(
                module=self.entrypoint_module, args=args, cwd=tmp, timeout_sec=self.timeout_sec
            )
            if proc.returncode != 0:
                return EvalResult(
                    resolved=False,
                    info={
                        "error": "harness_failed",
                        "returncode": proc.returncode,
                        "stderr_tail": _tail(proc.stderr or ""),
                        "stdout_tail": _tail(proc.stdout or ""),
                    },
                )

            # Persist full logs
            try:
                (work_out / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
                (work_out / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            except Exception:
                pass

            # Parse report
            resolved, info = _report_indicates_resolved(work_out)
            # Persist artifacts if requested output_dir is different from tmp
            try:
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    # copy tree (shallow)
                    for p in work_out.rglob("*"):
                        if p.is_file():
                            dst = self.output_dir / p.relative_to(work_out)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(p, dst)
            except Exception:
                pass
            return EvalResult(resolved=resolved, info=info)


class OpenHandsRunner:
    """Runner that shells to a user-provided MopenHands entrypoint.

    It mirrors HarnessRunner behavior but uses a caller-supplied module path.
    """

    def __init__(
        self,
        *,
        dataset_file: str,
        output_dir: str = "./msb_runs",
        # Preferred name matching docs/loader
        openhands_entrypoint: str | None = None,
        # Back-compat alias matching earlier call-sites
        entrypoint_module: str | None = None,
        timeout_sec: int = 1800,
        extra_args: list[str] | None = None,
    ) -> None:
        if not Path(dataset_file).is_file():
            raise ValueError(f"dataset_file does not exist or is not a file: {dataset_file}")
        self.dataset_file = dataset_file
        self.output_dir = Path(output_dir)
        self.entrypoint_module = (
            openhands_entrypoint or entrypoint_module or "mopenhands.run"
        )
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []

    def evaluate(self, *, instance_id: str, patch: str) -> EvalResult:
        with tempfile.TemporaryDirectory(prefix="msb_oh_") as tmp:
            tmpdir = Path(tmp)
            work_out = tmpdir / "out"
            work_out.mkdir(parents=True, exist_ok=True)

            patches_path = tmpdir / "patches.jsonl"
            _write_text(
                patches_path,
                json.dumps({"instance_id": instance_id, "patch": patch}) + "\n",
            )
            cfg = {
                "dataset_file": os.fspath(Path(self.dataset_file).resolve()),
                "patches_path": os.fspath(patches_path),
                "output_dir": os.fspath(work_out),
                "max_workers": 1,
                "instances": [instance_id],
            }
            cfg_path = tmpdir / "config.json"
            _write_text(cfg_path, json.dumps(cfg))

            args = ["--config", os.fspath(cfg_path), *self.extra_args]
            proc = _run_entrypoint(
                module=self.entrypoint_module, args=args, cwd=tmp, timeout_sec=self.timeout_sec
            )
            if proc.returncode != 0:
                return EvalResult(
                    resolved=False,
                    info={
                        "error": "openhands_failed",
                        "returncode": proc.returncode,
                        "stderr_tail": _tail(proc.stderr or ""),
                        "stdout_tail": _tail(proc.stdout or ""),
                    },
                )

            # Persist full logs
            try:
                (work_out / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
                (work_out / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            except Exception:
                pass

            resolved, info = _report_indicates_resolved(work_out)
            try:
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    for p in work_out.rglob("*"):
                        if p.is_file():
                            dst = self.output_dir / p.relative_to(work_out)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(p, dst)
            except Exception:
                pass
            return EvalResult(resolved=resolved, info=info)
