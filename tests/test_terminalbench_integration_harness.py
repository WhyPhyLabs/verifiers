import os
import pytest
import verifiers as vf
from datasets import Dataset


@pytest.mark.integration
def test_terminalbench_harness_integration(mock_openai_client):
    dataset_path = os.getenv("TB_DATASET_PATH")
    task_id = os.getenv("TB_TASK_ID")
    agent = os.getenv("TB_AGENT", "terminus")
    if not dataset_path or not task_id:
        pytest.skip("Set TB_DATASET_PATH and TB_TASK_ID to run Terminal-Bench harness integration")

    prompt_text = "Use the terminal to execute a simple command."
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": prompt_text}],
        response="echo hello",
    )
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt_text}]],
            "answer": ["OK"],
            "info": [{}],
            "task": [task_id],
        }
    )

    # Instantiate the harness runner directly to ensure harness code path
    runner = vf.integrations.terminalbench.HarnessRunner(
        dataset_path=dataset_path, agent_name=agent
    )
    env = vf.TerminalBenchEnv(runner=runner, task_id=task_id, eval_dataset=ds)
    env.rubric = vf.TerminalBenchRubric()
    res = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 32, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(res.reward) == 1

