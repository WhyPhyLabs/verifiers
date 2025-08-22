import verifiers as vf
from verifiers.integrations.multiswebench import StubRunner

try:
    from verifiers.integrations.multiswebench import HarnessRunner
except Exception:  # pragma: no cover - optional dependency
    HarnessRunner = None  # type: ignore


def load_environment(
    task_id: str = "stub",
    expected_token: str = "PASS_FIX",
    dataset_file: str | None = None,
    workdir: str | None = None,
    output_dir: str | None = None,
    repo_dir: str | None = None,
    max_workers: int = 1,
    force_build: bool = False,
    **kwargs,
) -> vf.Environment:
    """Load a Multi‑SWE‑Bench environment.

    Args:
        task_id: Benchmark task identifier.
        expected_token: For stub runner only; token that indicates a successful patch.
    """
    if dataset_file and HarnessRunner:
        runner = HarnessRunner(
            dataset_file=dataset_file,
            workdir=workdir,
            output_dir=output_dir,
            repo_dir=repo_dir,
            max_workers=max_workers,
            force_build=force_build,
        )
    else:
        runner = StubRunner(expected_token=expected_token)

    env = vf.MultiSWEEnv(runner=runner, task_id=task_id)
    env.rubric = vf.MultiSWERubric()  # simple pass/fail rubric
    return env
