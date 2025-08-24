# vf-multi-swe-bench

Multi‑SWE‑Bench adapter environment for verifiers. Wraps `MultiSWEEnv` and uses a
runner implementation to interact with the official harness or MopenHands.

Quickstart (stub runner):

```
uv run vf-eval vf-multi-swe-bench -a '{"instance_id":"stub-task","expected_token":"FIX_PATCH"}' -n 1 -r 1
```

Integration harness (optional)
------------------------------

See `docs/multiswebench_integration.md` for using the official harness or OpenHands.
Required environment variables:

| Variable | Meaning |
|---------|---------|
| `MSB_DATASET_FILE` | Path to Multi‑SWE‑Bench JSONL shard |
| `MSB_INSTANCE_ID` or `MSB_TASK_ID` | Instance/task id `<org>__<repo>-<number>` |
| `OPENHANDS_ENTRYPOINT` | Module path to OpenHands entrypoint (when `use_openhands=true`) |
