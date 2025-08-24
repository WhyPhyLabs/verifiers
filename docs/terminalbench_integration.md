# Terminal-Bench Integration

This repository includes a production-ready adapter for Terminal-Bench tasks.

- Env: `verifiers.envs.TerminalBenchEnv`
- Runner interface: `verifiers.integrations.terminalbench.TerminalBenchRunner`
- Stub runner (for CI): `StubRunner` (no external deps)
- Harness runner: `HarnessRunner` (uses official `terminal_bench` Python API)
- Rubric: `TerminalBenchRubric` (1.0 pass, 0.0 fail)

Quick usage with stub runner:

```
uv run vf-eval vf-terminal-bench -a '{"task_id":"stub","expected_command":"echo hello"}' -n 1 -r 1
```

Production usage (official harness):

```
uv run vf-eval vf-terminal-bench \
  -a '{"task_id":"word2vec-from-scratch","dataset_path":"/path/to/terminal-bench/tasks"}' \
  -n 1 -r 1
```

Notes:
- Ensure Docker is available and `terminal_bench` is installed.
- The environment will use `HarnessRunner` when `dataset_path` is provided; otherwise it
  falls back to `StubRunner`.

Custom integration:
- You can implement your own `TerminalBenchRunner` and inject it into `TerminalBenchEnv`.

Testing:
- Smoke test: `tests/test_terminalbench_env.py` validates end-to-end flow with `MockOpenAIClient`.
- Run: `uv run pytest -k terminalbench -q`.
