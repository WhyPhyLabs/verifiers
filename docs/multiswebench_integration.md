# Multi‑SWE‑Bench Integration

This repository includes an adapter for evaluating Multi‑SWE‑Bench instances
with either the official harness or the MopenHands harness.

- Env: `verifiers.envs.MultiSWEEnv`
- Runner interface: `verifiers.integrations.multiswebench.MultiSWERunner`
- Runners: `HarnessRunner` (official), `OpenHandsRunner` (MopenHands), `StubRunner`
- Rubric: `MultiSWERubric` (1.0 when resolved, else 0.0)

Quick usage (stub):

```
uv run vf-eval vf-multi-swe-bench -a '{"instance_id":"example"}' -n 1 -r 1
```

Official harness (single‑instance):

```
uv run vf-eval vf-multi-swe-bench \
  -a '{"instance_id":"<ID>", "dataset_file":"/path/to/dataset.jsonl"}' \
  -n 1 -r 1
```

MopenHands harness:

```
uv run vf-eval vf-multi-swe-bench \
  -a '{"instance_id":"<ID>", "dataset_file":"/path/to/dataset.jsonl", "use_openhands":true, "openhands_entrypoint":"mopenhands.run"}' \
  -n 1 -r 1
```

Notes:
- Both harness runners use `sys.executable -m <module> --config <cfg.json>` under the hood and expect the harness to write a `final_report.json` in the output directory. The adapter falls back to `report.json` when needed.
- Timeouts and return‑code checks are enabled; failures surface stderr/stdout tails in the result info for debugging.
- The environment is single‑turn by default: the model’s last message is treated as the candidate patch.

Testing:
- Smoke test: add `-k multiswe` to run stub‑based tests, or set env‑vars to opt‑in to integration tests in your CI.
