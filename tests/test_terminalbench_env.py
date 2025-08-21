import verifiers as vf
from verifiers.integrations.terminalbench import StubRunner
from datasets import Dataset


def test_terminalbench_env_smoke_chat(mock_openai_client):
    # Async client responds with the expected command when it sees the stub prompt
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": "Use the terminal to say hello."}],
        response="echo hello",
    )
    rubric = vf.TerminalBenchRubric()
    dataset_dict = {
        "prompt": [[{"role": "user", "content": "Use the terminal to say hello."}]],
        "answer": ["OK"],
        "info": [{}],
        "task": ["terminal-stub"],
    }
    ds = Dataset.from_dict(dataset_dict)
    env = vf.TerminalBenchEnv(
        runner=StubRunner(expected_command="echo hello"),
        task_id="stub-task",
        eval_dataset=ds,
    )
    env.rubric = rubric
    results = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 64, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(results.reward) == 1
    assert results.reward[0] == 1.0
