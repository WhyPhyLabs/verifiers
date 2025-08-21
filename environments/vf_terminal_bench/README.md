# vf-terminal-bench

Terminal-Bench adapter environment for verifiers. Wraps the TerminalBenchEnv and uses a runner implementation to interact with the Terminal-Bench harness.

Quickstart:

```
uv run vf-eval vf-terminal-bench -a '{"task_id":"stub-task","expected_command":"echo hello"}' -n 1 -r 1
```
