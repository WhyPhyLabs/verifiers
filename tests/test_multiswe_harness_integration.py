import os
import pathlib
import pytest

import verifiers as vf


@pytest.mark.integration
def test_multiswe_harness_runner_smoke(monkeypatch):
    dataset_file = os.environ.get("MSB_DATASET_FILE")
    task_id = os.environ.get("MSB_TASK_ID")
    if not dataset_file or not task_id:
        pytest.skip("Set MSB_DATASET_FILE and MSB_TASK_ID to run harness smoke test")

    # Ensure file exists
    assert pathlib.Path(dataset_file).exists()

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

    # Use a trivial patch to drive the loop; likely fails but should run the pipeline
    prompt = [[{"role": "user", "content": "Propose a patch."}]]
    client = object()  # not used by env; driver uses messages directly
    # Use the ground-truth fix patch if available to maximize chance of success
    import json
    fix_patch = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@\n-foo\n+bar\n\n"
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            inst_id = f"{item['org']}__{item['repo']}-{item['number']}"
            if inst_id == task_id and item.get("fix_patch"):
                fix_patch = item["fix_patch"]
                break

    results = env.evaluate(
        client=client,
        model="ignored",
        sampling_args={"max_tokens": 1},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent_requests=1,
        driver=lambda *_args, **_kwargs: [{"role": "assistant", "content": fix_patch}],
    )
    assert len(results.reward) == 1
