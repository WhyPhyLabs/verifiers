import verifiers as vf

try:
    from verifiers.integrations.multiswebench import HarnessRunner, OpenHandsRunner
except Exception:  # pragma: no cover - optional dependency
    HarnessRunner = None  # type: ignore
    OpenHandsRunner = None  # type: ignore


def load_environment(
    instance_id: str | None = "example-instance",
    *,
    task_id: str | None = None,
    dataset_file: str | None = None,
    output_dir: str | None = None,
    use_openhands: bool | None = None,
    openhands_entrypoint: str | None = None,
    eval_dataset=None,
    **kwargs,
) -> vf.Environment:
    """Load a Multi-SWE-Bench single-instance environment.

    Args:
        instance_id: Identifier of the dataset instance to evaluate.
        task_id: Back-compat alias for instance_id.
        dataset_file: Path to the Multi-SWE-Bench dataset JSONL file.
        output_dir: Directory to store harness artifacts (default ./msb_runs).
        use_openhands: When True, use the MopenHands harness via `openhands_entrypoint`.
        openhands_entrypoint: Module path to the MopenHands entrypoint (e.g.,
            "mopenhands.run"). If None, a sensible default is used by the runner.
        eval_dataset: Optional HF Dataset to drive prompts/answers.
    """
    runner = None
    out = output_dir or "./msb_runs"
    inst_id = instance_id or task_id or "example-instance"

    if use_openhands and OpenHandsRunner and dataset_file:
        runner = OpenHandsRunner(dataset_file=dataset_file, output_dir=out, openhands_entrypoint=openhands_entrypoint or "mopenhands.run")
    elif dataset_file and HarnessRunner:
        runner = HarnessRunner(dataset_file=dataset_file, output_dir=out)
    else:
        from verifiers.integrations.multiswebench import StubRunner

        runner = StubRunner()

    if eval_dataset is None:
        from datasets import Dataset

        instruction = (
            "Produce a unified diff patch that resolves the failing tests for the given instance. "
            "Include only the patch text."
        )
        eval_dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": instruction}]],
                "answer": ["resolved"],
                "info": [{}],
                "task": [inst_id],
            }
        )

    env = vf.MultiSWEEnv(runner=runner, instance_id=inst_id, eval_dataset=eval_dataset)
    env.rubric = vf.MultiSWERubric()
    return env
