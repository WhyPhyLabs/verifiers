import verifiers as vf
from verifiers.integrations.multiswebench import StubRunner


def load_environment(task_id: str = "stub", expected_token: str = "PASS_FIX", **kwargs) -> vf.Environment:  # noqa: ARG001
    """Load a Multi‑SWE‑Bench environment.

    Args:
        task_id: Benchmark task identifier.
        expected_token: For stub runner only; token that indicates a successful patch.
    """
    runner = StubRunner(expected_token=expected_token)
    env = vf.MultiSWEEnv(runner=runner, task_id=task_id)
    env.rubric = vf.MultiSWERubric()  # simple pass/fail rubric
    return env
