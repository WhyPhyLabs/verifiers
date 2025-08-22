"""Multi-SWE-Bench integration shims.

Defines a lightweight runner protocol to integrate Multi-SWE-Bench style
program-repair tasks with the verifiers Environment API without hard deps.

Production deployments should provide a concrete runner that talks to the
official Multi-SWE-Bench harness to checkout repos, apply patches, run tests,
and decide pass/fail. This package also includes a `StubRunner` suitable for
CI smoke tests.
"""

from .runner import (
    MultiSWERunner,
    PatchStepResult,
    StubRunner,
    HarnessRunner,
)  # noqa: F401
