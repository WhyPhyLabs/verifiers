"""Terminal-Bench integration shims.

This package defines lightweight interfaces to integrate Terminal-Bench tasks
with the verifiers Environment API without imposing hard dependencies.

Production deployments should provide a concrete runner implementation that
speaks to the Terminal-Bench harness (e.g., via MCP, Python API, or REST).
"""

from .runner import TerminalBenchRunner, StubRunner, StepResult  # noqa: F401
