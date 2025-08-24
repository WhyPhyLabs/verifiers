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

    # Provide a minimal single-example eval dataset if not supplied,
    # so the environment can run directly via `vf-eval`.
    eval_dataset = kwargs.get("eval_dataset")
    if eval_dataset is None:
        from datasets import Dataset

        instruction = (
            "Propose a unified diff patch to fix the failing tests. "
            "Respond with only the patch."
        )
        eval_dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": instruction}]],
                "answer": ["OK"],
                "info": [{}],
                "task": [task_id],
            }
        )

    env = vf.MultiSWEEnv(runner=runner, task_id=task_id, eval_dataset=eval_dataset)
    env.rubric = vf.MultiSWERubric()  # simple pass/fail rubric
    return env
