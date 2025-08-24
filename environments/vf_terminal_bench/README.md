# vf-terminal-bench

Terminal-Bench adapter environment for verifiers. Wraps the TerminalBenchEnv and uses a runner implementation to interact with the Terminal-Bench harness.

Quickstart:

```
uv run vf-eval vf-terminal-bench -a '{"task_id":"stub-task","expected_command":"echo hello"}' -n 1 -r 1
```

Integration harness (optional)
------------------------------

When using the official Terminal‑Bench harness, see `docs/terminalbench_integration.md`.
Required environment variables:

| Variable | Meaning |
|---------|---------|
| `TB_DATASET_PATH` | Path to Terminal‑Bench tasks directory |
| `TB_TASK_ID`      | Task id |
| `TB_ENTRYPOINT`   | Harness entrypoint module (e.g., `terminal_bench.run`) |
