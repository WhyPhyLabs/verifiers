import verifiers as vf
from verifiers.integrations.multiswebench import StubRunner
from datasets import Dataset


def test_multiswe_env_smoke_chat(mock_openai_client):
    # Async client proposes a patch containing the expected token
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": "Propose a patch to fix tests."}],
        response="""--- a/module.py
+++ b/module.py
@@
- return 0
+ return 1  # PASS_FIX
""",
    )
    rubric = vf.MultiSWERubric()
    dataset_dict = {
        "prompt": [[{"role": "user", "content": "Propose a patch to fix tests."}]],
        "answer": ["OK"],
        "info": [{}],
        "task": ["multiswe-stub"],
    }
    ds = Dataset.from_dict(dataset_dict)
    env = vf.MultiSWEEnv(
        runner=StubRunner(expected_token="PASS_FIX"),
        task_id="stub-task",
        eval_dataset=ds,
    )
    env.rubric = rubric
    results = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 128, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(results.reward) == 1
    assert results.reward[0] == 1.0
