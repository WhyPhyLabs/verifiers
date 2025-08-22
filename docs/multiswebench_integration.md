# Multi‑SWE‑Bench Integration

This repository includes a thin adapter for Multi‑SWE‑Bench style program repair tasks.

- Env: `verifiers.envs.MultiSWEEnv`
- Runner interface: `verifiers.integrations.multiswebench.MultiSWERunner`
- Stub runner (for CI): `StubRunner` (no external deps)
- Harness runner: `HarnessRunner` (uses official `multi_swe_bench` Python API)
- Rubric: `MultiSWERubric` (1.0 pass, 0.0 fail)

Quick usage with stub runner:

```
uv run vf-eval vf-multi-swe-bench -a '{"task_id":"stub","expected_token":"PASS_FIX"}' -n 1 -r 1
```

Production usage:
- Use the built-in `HarnessRunner` by supplying a dataset file and task id.

Example (single instance, stub OpenAI for model calls):

```
uv run vf-install vf-multi-swe-bench
uv run vf-eval vf-multi-swe-bench \
  -a '{
        "task_id":"axios__axios-5919",
        "dataset_file":"/mnt/beegfs/agents/model_repository/datasets/multi-swe-bench/js/axios__axios_dataset.jsonl",
        "workdir":"./msb_work",
        "output_dir":"./msb_out",
        "repo_dir":"./msb_repos",
        "max_workers":1
      }' \
  -n 1 -r 1 \
  -m mock -b http://127.0.0.1:8009/v1 -k OPENAI_API_KEY
```

Notes:
- Ensure Docker is available. First-run will build environment images and can take time.
- `task_id` format is `<org>__<repo>-<number>` matching dataset entries.

Testing:
- Smoke test: `tests/test_multiswe_env.py` validates end-to-end flow with `mock_openai_client` and `StubRunner`.
- Run: `uv run pytest -k multiswe -q`.
