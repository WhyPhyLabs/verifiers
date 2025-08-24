"""Multi-SWE-Bench integration shims.

This package provides thin, optional integrations to run Multi-SWE-Bench
evaluations from the verifiers environments without imposing hard deps.

It exposes:
- Runner protocol `MultiSWERunner`
- `StubRunner` for CI smoke
- `HarnessRunner` that shells to the official harness
- `OpenHandsRunner` that shells to a provided entrypoint for the MopenHands harness

Both concrete runners are intentionally entrypoint-parameterized and guarded so
users can opt-in when the respective packages are installed.
"""

from .runner import (  # noqa: F401
    MultiSWERunner,
    StubRunner,
    HarnessRunner,
    OpenHandsRunner,
    EvalResult,
)

