# Terminal-Bench Integration

This repository includes a production-ready adapter for Terminal-Bench tasks.

- Env: `verifiers.envs.TerminalBenchEnv`
- Runner interface: `verifiers.integrations.terminalbench.TerminalBenchRunner`
- Stub runner (for CI): `StubRunner` (no external deps)
- Rubric: `TerminalBenchRubric` (1.0 pass, 0.0 fail)
  
This adapter ships a runner Protocol, a local StubRunner, and a HarnessRunner that
drives the official Terminal‑Bench harness via a configurable entrypoint module
(`terminal_bench.run` by default). The harness path uses `sys.executable -m`,
adds timeouts/return‑code checks, persists logs in `output_path`, and defaults
the agent to `terminus` (matching upstream `tb run --agent terminus`).

Quick usage with stub runner:

```
uv run vf-eval vf-terminal-bench -a '{"task_id":"stub","expected_command":"echo hello"}' -n 1 -r 1
```

Production usage (official harness):

```
uv run vf-eval vf-terminal-bench \
  -a '{"task_id":"word2vec-from-scratch","dataset_path":"/path/to/terminal-bench/tasks","use_harness":true}' \
  -n 1 -r 1
```

Notes:
- Ensure Docker is available and `terminal_bench` is installed.
- The example environment will automatically select the HarnessRunner when
  `dataset_path` is provided (or `use_harness=true`), falling back to `StubRunner`
  otherwise.
  
Advanced:
- You can override the entrypoint via `entrypoint_module` and pass additional
  flags using `extra_args`.

Custom integration:
- You can implement your own `TerminalBenchRunner` and inject it into `TerminalBenchEnv`.

Testing:
- Smoke test: `tests/test_terminalbench_env.py` validates end-to-end flow with `MockOpenAIClient`.
- Run: `uv run pytest -k terminalbench -q`.

Implementation detail:
- The environment seeds the initial terminal observation (from `runner.start`) as a tool
  message before the first model turn, so the model immediately sees the prompt/shell context.

## Environment variables (for harness runs and integration tests)

| Variable | Required | Description |
|---------|----------|-------------|
| `TB_DATASET_PATH` | Yes (harness) | Path to Terminal‑Bench tasks directory |
| `TB_TASK_ID` | Yes | Task id |
| `TB_ENTRYPOINT` | Yes (harness) | Harness entrypoint module (e.g., `terminal_bench.run`) |
