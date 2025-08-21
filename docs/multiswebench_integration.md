# Multi‑SWE‑Bench Integration

This repository includes a thin adapter for Multi‑SWE‑Bench style program repair tasks.

- Env: `verifiers.envs.MultiSWEEnv`
- Runner interface: `verifiers.integrations.multiswebench.MultiSWERunner`
- Stub runner (for CI): `StubRunner` (no external deps)
- Rubric: `MultiSWERubric` (1.0 pass, 0.0 fail)

Quick usage with stub runner:

```
uv run vf-eval vf-multi-swe-bench -a '{"task_id":"stub","expected_token":"PASS_FIX"}' -n 1 -r 1
```

Production usage:
- Implement a `MultiSWERunner` using the official harness to checkout repos, apply patches, run tests, and return pass/fail.
- Inject your runner into `vf-multi-swe-bench` or your own environment module.
- Use `vf-eval` for evaluations and `report_utils` for HTML reports.

Testing:
- Smoke test: `tests/test_multiswe_env.py` validates end-to-end flow with `mock_openai_client` and `StubRunner`.
- Run: `uv run pytest -k multiswe -q`.
