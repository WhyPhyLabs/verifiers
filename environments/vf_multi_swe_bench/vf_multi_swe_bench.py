import verifiers as vf
from verifiers.integrations.multiswebench import StubRunner

try:
    from verifiers.integrations.multiswebench import HarnessRunner, OpenHandsRunner
except Exception:  # pragma: no cover - optional dependency
    HarnessRunner = None  # type: ignore
    OpenHandsRunner = None  # type: ignore


def load_environment(
    task_id: str = "stub",
    expected_token: str = "PASS_FIX",
    dataset_file: str | None = None,
    workdir: str | None = None,
    output_dir: str | None = None,
    repo_dir: str | None = None,
    max_workers: int = 1,
    force_build: bool = False,
    use_openhands: bool | None = None,
    openhands_entrypoint: str | None = None,
    openhands_args: list[str] | None = None,
    eval_dataset=None,
    **kwargs,
) -> vf.Environment:
    """Load a Multi‑SWE‑Bench environment.

    Selection logic (in order):
    - If use_openhands and OpenHandsRunner is available: use OpenHands runner (requires entrypoint)
    - Else if dataset_file and HarnessRunner is available: use official harness runner
    - Else: use StubRunner
    """
    if use_openhands and OpenHandsRunner:
        if not dataset_file or not openhands_entrypoint:
            raise ValueError("OpenHandsRunner requires dataset_file and openhands_entrypoint")
        runner = OpenHandsRunner(
            dataset_file=dataset_file,
            entrypoint_module=openhands_entrypoint,
            entrypoint_args=openhands_args or [],
            workdir=workdir,
            output_dir=output_dir,
            repo_dir=repo_dir,
            max_workers=max_workers,
            force_build=force_build,
        )
    elif dataset_file and HarnessRunner:
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
    env.rubric = vf.MultiSWERubric()
    return env
