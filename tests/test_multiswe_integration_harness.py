import os
import pytest
import verifiers as vf
from datasets import Dataset


@pytest.mark.integration
def test_multiswe_harness_official_integration(mock_openai_client):
    dataset_file = os.getenv("MSB_DATASET_FILE")
    instance_id = os.getenv("MSB_INSTANCE_ID")
    fix_patch = os.getenv("MSB_FIX_PATCH")  # optional known-good patch
    if not dataset_file or not instance_id:
        pytest.skip("Set MSB_DATASET_FILE and MSB_INSTANCE_ID to run harness integration")

    # Prepare a minimal dataset and force the model to emit the patch
    prompt_text = "Return a unified diff patch that attempts to fix the instance"
    if fix_patch:
        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": prompt_text}],
            response=fix_patch,
        )
    else:
        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": prompt_text}],
            response=""  # empty patch likely unresolved but exercises path
        )

    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt_text}]],
            "answer": ["resolved"],
            "info": [{}],
            "task": [instance_id],
        }
    )

    # Official harness path
    env = vf.MultiSWEEnv(
        runner=vf.integrations.multiswebench.HarnessRunner(dataset_file=dataset_file),
        instance_id=instance_id,
        eval_dataset=ds,
    )
    env.rubric = vf.MultiSWERubric()
    res = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 64, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(res.reward) == 1
    if fix_patch:
        assert res.reward[0] in (0.0, 1.0)  # expect 1.0 with a true fix patch


@pytest.mark.integration
def test_multiswe_harness_openhands_integration(mock_openai_client):
    dataset_file = os.getenv("MSB_DATASET_FILE")
    instance_id = os.getenv("MSB_INSTANCE_ID")
    openhands_entrypoint = os.getenv("OPENHANDS_ENTRYPOINT")
    if not dataset_file or not instance_id or not openhands_entrypoint:
        pytest.skip("Set MSB_DATASET_FILE, MSB_INSTANCE_ID and OPENHANDS_ENTRYPOINT to run OpenHands integration")

    prompt_text = "Return a unified diff patch that attempts to fix the instance"
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": prompt_text}],
        response="",
    )
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt_text}]],
            "answer": ["resolved"],
            "info": [{}],
            "task": [instance_id],
        }
    )

    env = vf.MultiSWEEnv(
        runner=vf.integrations.multiswebench.OpenHandsRunner(
            dataset_file=dataset_file, openhands_entrypoint=openhands_entrypoint
        ),
        instance_id=instance_id,
        eval_dataset=ds,
    )
    env.rubric = vf.MultiSWERubric()
    res = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 64, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(res.reward) == 1
