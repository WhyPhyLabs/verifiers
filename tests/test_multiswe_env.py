import verifiers as vf
from verifiers.integrations.multiswebench import StubRunner
from datasets import Dataset


def test_multiswe_env_stub_smoke(mock_openai_client):
    # The agent is prompted to output a patch containing FIX_PATCH to pass
    user_prompt = "Return a patch that contains the token FIX_PATCH"
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": user_prompt}],
        response="""
--- a/foo.py
+++ b/foo.py
@@
# FIX_PATCH
""".strip(),
    )
    rubric = vf.MultiSWERubric()
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": user_prompt}]],
            "answer": ["resolved"],
            "info": [{}],
            "task": ["example-instance"],
        }
    )
    env = vf.MultiSWEEnv(runner=StubRunner(), instance_id="example-instance", eval_dataset=ds)
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
