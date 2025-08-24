import os
import json
import pathlib
import pytest
from datasets import Dataset

import verifiers as vf


@pytest.mark.integration
def test_multiswe_openhands_runner_smoke(mock_openai_client):
    dataset_file = os.environ.get("MSB_DATASET_FILE")
    task_id = os.environ.get("MSB_TASK_ID")
    entrypoint = os.environ.get("OPENHANDS_ENTRYPOINT")
    if not dataset_file or not task_id or not entrypoint:
        pytest.skip("Set MSB_DATASET_FILE, MSB_TASK_ID, and OPENHANDS_ENTRYPOINT to run OpenHands smoke test")

    assert pathlib.Path(dataset_file).exists()

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
        use_openhands=True,
        openhands_entrypoint=entrypoint,
        workdir="./msb_work_test_oh",
        output_dir="./msb_out_test_oh",
        repo_dir="./msb_repos_test_oh",
        max_workers=1,
        force_build=False,
    )
    env.rubric = vf.MultiSWERubric()

    # Use dataset fix_patch if present
    fix_patch = None
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            inst_id = f"{item['org']}__{item['repo']}-{item['number']}"
            if inst_id == task_id and item.get("fix_patch"):
                fix_patch = item["fix_patch"]
                break

    if fix_patch:
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
