import verifiers as vf
from verifiers.integrations.terminalbench import StubRunner

try:
    # Optional import; only used when dataset_path is provided and harness is desired
    from verifiers.integrations.terminalbench import HarnessRunner  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    HarnessRunner = None  # type: ignore


def load_environment(
    task_id: str = "stub",
    expected_command: str = "echo hello",
    dataset_path: str | None = None,
    output_path: str | None = None,
    use_harness: bool | None = None,
    eval_dataset=None,
    **kwargs,
) -> vf.Environment:
    """Load a Terminal-Bench environment.

    Args:
        task_id: Terminal-Bench task identifier.
        expected_command: For stub runner only; command that passes the task.
        dataset_path: Path to terminal-bench tasks directory. When provided and
            a harness is installed, the official harness can be used.
        output_path: Output directory for harness artifacts (defaults to ./tb_runs).
        use_harness: Force using the harness when True, otherwise auto-detect based
            on dataset_path availability.
    """
    # Fail fast: if harness is explicitly requested, require dataset_path
    if use_harness and not dataset_path:
        raise ValueError("use_harness=True requires dataset_path to be provided")

    if (use_harness or (use_harness is None and dataset_path)) and HarnessRunner:
        runner = HarnessRunner(  # type: ignore[call-arg]
            dataset_path=dataset_path,  # type: ignore[arg-type]
            output_path=output_path or "./tb_runs",
        )
    else:
        runner = StubRunner(expected_command=expected_command)

    # Provide a minimal single-example eval dataset if one is not supplied,
    # so the environment can be used directly via `vf-eval`.
    if eval_dataset is None:
        from datasets import Dataset

        instruction = (
            f"Use the terminal to solve task '{task_id}'. "
            "Respond with only the exact shell command to run."
        )
        eval_dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": instruction}]],
                "answer": ["OK"],
                "info": [{}],
                "task": [task_id],
            }
        )

    env = vf.TerminalBenchEnv(runner=runner, task_id=task_id, eval_dataset=eval_dataset)
    env.rubric = vf.TerminalBenchRubric()  # attach simple pass/fail rubric
    return env
