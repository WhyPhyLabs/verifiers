import os
import json
import pathlib
import pytest
from datasets import Dataset

import verifiers as vf


@pytest.mark.integration
def test_multiswe_harness_runner_smoke(mock_openai_client):
    dataset_file = os.environ.get("MSB_DATASET_FILE")
    task_id = os.environ.get("MSB_TASK_ID")
    if not dataset_file or not task_id:
        pytest.skip("Set MSB_DATASET_FILE and MSB_TASK_ID to run harness smoke test")

    # Ensure file exists
    assert pathlib.Path(dataset_file).exists()

    # Prepare a minimal single-example dataset driving one patch attempt
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Propose a patch to fix tests."}]],
            "answer": ["OK"],
            "info": [{}],
            "task": [task_id],
        }
    )

    env = vf.load_environment(
        env_id="vf-multi-swe-bench",
        task_id=task_id,
        dataset_file=dataset_file,
        workdir="./msb_work_test",
        output_dir="./msb_out_test",
        repo_dir="./msb_repos_test",
        max_workers=1,
        force_build=False,
    )
    env.rubric = vf.MultiSWERubric()

    # Ground-truth patch to maximize success likelihood
    fix_patch = None
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            inst_id = f"{item['org']}__{item['repo']}-{item['number']}"
            if inst_id == task_id and item.get("fix_patch"):
                fix_patch = item["fix_patch"]
                break
    assert fix_patch, "No fix_patch found in dataset for the specified task"

    # Stub the model to produce the exact fix_patch at first turn
    mock_openai_client.add_chat_response(
        messages=[{"role": "user", "content": "Propose a patch to fix tests."}],
        response=fix_patch,
    )

    env.eval_dataset = ds
    results = env.evaluate(
        client=mock_openai_client,
        model="dummy-chat",
        sampling_args={"max_tokens": 2048, "temperature": 0.0},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )
    assert len(results.reward) == 1
