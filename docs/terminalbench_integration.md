# Terminal-Bench Integration

This repository includes a production-ready adapter for Terminal-Bench tasks.

- Env: `verifiers.envs.TerminalBenchEnv`
- Runner interface: `verifiers.integrations.terminalbench.TerminalBenchRunner`
- Stub runner (for CI): `StubRunner` (no external deps)
- Rubric: `TerminalBenchRubric` (1.0 pass, 0.0 fail)

Quick usage with stub runner:

```
uv run vf-eval vf-terminal-bench -a '{"task_id":"stub","expected_command":"echo hello"}' -n 1 -r 1
```

Production usage:
- Implement a `TerminalBenchRunner` using the official harness (MCP, Python, or REST).
- Inject your runner into `vf-terminal-bench` or your own environment module.
- Use `vf-eval` for evaluations and `report_utils` for HTML reports.

Testing:
- Smoke test: `tests/test_terminalbench_env.py` validates end-to-end flow with `MockOpenAIClient`.
- Run: `uv run pytest -k terminalbench -q`.

## Environment variables (for harness runs and integration tests)

| Variable | Required | Description |
|---------|----------|-------------|
| `TB_DATASET_PATH` | Yes (harness) | Path to Terminal‑Bench tasks directory |
| `TB_TASK_ID` | Yes | Task id |
| `TB_ENTRYPOINT` | Yes (harness) | Harness entrypoint module (e.g., `terminal_bench.run`) |

Notes
- When `use_harness=true`, you must provide `dataset_path`. The loader fails fast
  with a clear error if `use_harness=true` and `dataset_path` is missing.
- The harness runner enforces a default timeout of 1800s; for programmatic use of
  the runner class you can override `timeout_sec` if your hardware is slower.
