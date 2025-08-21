import verifiers as vf
from verifiers.integrations.terminalbench import StubRunner


def load_environment(task_id: str = "stub", expected_command: str = "echo hello", **kwargs) -> vf.Environment:  # noqa: ARG001
    """Load a Terminal-Bench environment.

    Args:
        task_id: Terminal-Bench task identifier.
        expected_command: For stub runner only; command that passes the task.
    """
    runner = StubRunner(expected_command=expected_command)
    env = vf.TerminalBenchEnv(runner=runner, task_id=task_id)
    env.rubric = vf.TerminalBenchRubric()  # attach simple pass/fail rubric
    return env
