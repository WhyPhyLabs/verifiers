from __future__ import annotations

from typing import Any

from .multiturn_env import MultiTurnEnv
from ..integrations.multiswebench import EvalResult, MultiSWERunner
from ..types import Messages, State


class MultiSWEEnv(MultiTurnEnv):
    """Environment adapter for Multi-SWE-Bench single-instance evaluation.

    The model is expected to produce a patch (e.g., unified diff) as plain text.
    On each turn we extract the latest assistant message as the candidate patch
    and ask the configured runner to evaluate it for the selected instance.

    A minimal rubric can then reward success (resolved=True) with 1.0.
    """

    def __init__(
        self,
        runner: MultiSWERunner,
        instance_id: str | None = None,
        *,
        task_id: str | None = None,
        max_turns: int = 1,
        **kwargs: Any,
    ) -> None:
        """Create a Multi‑SWE‑Bench environment.

        Args:
            runner: Concrete runner (official harness, OpenHands, or stub).
            instance_id: Identifier of the dataset instance (preferred).
            task_id: Back‑compat alias for instance_id.
            max_turns: Max turns (defaults to single‑turn evaluation).
        """
        super().__init__(**kwargs)
        self.runner = runner
        self.instance_id = instance_id or task_id or "example-instance"
        self.max_turns = max_turns

    def setup_state(self, state: State | None = None) -> State:
        st: State = state or {}
        st.setdefault("turn", 0)
        st.setdefault("done", False)
        st.setdefault("passed", False)
        return st

    def _extract_patch(self, messages: Messages) -> str:
        """Return a unified-diff patch string from the assistant's latest message.

        This is a light heuristic that treats the last assistant message content
        as the candidate patch. Behavior unchanged; documents the expectation.
        """
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            content = str(messages)
        return str(content).strip()

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:  # noqa: ARG002
        return bool(state.get("done") or state.get("turn", 0) >= self.max_turns)

    def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        st = self.setup_state(state)
        st["turn"] = st.get("turn", 0) + 1
        patch = self._extract_patch(messages)
        result: EvalResult = self.runner.evaluate(instance_id=self.instance_id, patch=patch)
        passed = bool(result.resolved)
        st.update({"done": True, "passed": passed, "report": result.info})
        # No additional tool output; simply echo a brief status for traceability
        obs = "resolved" if passed else "unresolved"
        msgs: list[dict[str, Any]]
        if isinstance(messages, list):
            msgs = list(messages)
        else:
            msgs = [{"role": "assistant", "content": str(messages)}]
        msgs.append({"role": "tool", "name": "msb", "content": obs})
        return msgs, st
