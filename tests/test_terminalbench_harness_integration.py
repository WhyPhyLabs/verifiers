import os
import pytest
from datasets import Dataset

import verifiers as vf


@pytest.mark.integration
def test_terminalbench_harness_runner_smoke(mock_openai_client):
    dataset_path = os.environ.get("TB_DATASET_PATH")
    task_id = os.environ.get("TB_TASK_ID")
    entrypoint = os.environ.get("TB_ENTRYPOINT")  # e.g., terminal_bench.run
    if not dataset_path or not task_id or not entrypoint:
        pytest.skip("Set TB_DATASET_PATH, TB_TASK_ID, and TB_ENTRYPOINT to run TB smoke test")

    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Use the terminal to solve the task."}]],
            "answer": ["OK"],
            "info": [{}],
            "task": [task_id],
        }
    )

    env = vf.load_environment(
        env_id="vf-terminal-bench",
        task_id=task_id,
        dataset_path=dataset_path,
        output_path="./tb_runs_test",
        use_harness=True,
    )
    env.rubric = vf.TerminalBenchRubric()

    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": "Use the terminal to solve the task."}],
        response="echo hello",
    )

    env.eval_dataset = ds
    results = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 128, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(results.reward) == 1
