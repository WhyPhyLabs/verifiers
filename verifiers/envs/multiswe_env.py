from __future__ import annotations

from typing import Any

from .multiturn_env import MultiTurnEnv
from ..integrations.multiswebench import MultiSWERunner, PatchStepResult
from ..types import Messages, State


class MultiSWEEnv(MultiTurnEnv):
    """Environment adapter for Multi‑SWE‑Bench style program repair tasks.

    Each assistant message is interpreted as a proposed patch (e.g., unified
    diff). The runner applies the patch and runs tests. The resulting
    observation (test output summary) is appended as a tool message.
    Completion is signaled by the runner when tests pass or attempts are
    exhausted.
    """

    def __init__(
        self,
        runner: MultiSWERunner,
        task_id: str,
        max_turns: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.runner = runner
        self.task_id = task_id
        self.max_turns = max_turns

    def setup_state(self, state: State | None = None) -> State:
        st: State = state or {}
        if st.get("_init_done"):
            return st
        initial_obs = self.runner.start(self.task_id)
        st.update(
            {
                "_init_done": True,
                "turn": 0,
                "observation": initial_obs,
                "done": False,
                "passed": False,
            }
        )
        return st

    def _extract_patch(self, messages: Messages) -> str:
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            content = str(messages)
        return str(content)

    def _append_tool_observation(self, messages: list[dict[str, Any]], obs: str) -> None:
        messages.append({"role": "tool", "content": obs, "name": "tests"})

    def is_completed(self, messages: Messages, state: State) -> bool:  # noqa: ARG002
        return bool(state.get("done") or state.get("turn", 0) >= self.max_turns)

    def env_response(self, messages: Messages, state: State) -> tuple[Messages, State]:
        st = self.setup_state(state)
        st["turn"] = st.get("turn", 0) + 1
        patch_text = self._extract_patch(messages)
        result: PatchStepResult = self.runner.apply_patch(patch_text)
        st.update({"done": result.done, "passed": result.passed, "observation": result.observation})
        # Clone messages to list for augmentation
        msgs: list[dict[str, Any]]
        if isinstance(messages, list):
            msgs = list(messages)
        else:
            msgs = [{"role": "user", "content": str(messages)}]
        self._append_tool_observation(msgs, result.observation)
        return msgs, st
