# vf-multi-swe-bench

Multi‑SWE‑Bench adapter environment for verifiers. Wraps the `MultiSWEEnv` and
uses a runner implementation to interact with the Multi‑SWE‑Bench harness.

Quickstart (stub runner):

```
uv run vf-eval vf-multi-swe-bench -a '{"task_id":"stub-task","expected_token":"PASS_FIX"}' -n 1 -r 1
```
