from __future__ import annotations

from typing import Any

from .multiturn_env import MultiTurnEnv
from ..integrations.terminalbench import StepResult, TerminalBenchRunner
from ..types import Messages, State


class TerminalBenchEnv(MultiTurnEnv):
    """Environment adapter for Terminal-Bench tasks.

    This environment delegates task execution to a provided `TerminalBenchRunner`
    instance. Each assistant message is interpreted as a shell command; the
    runner executes the command and returns an observation that is appended as a
    tool message for the next turn. Completion is signaled by the runner.
    """

    def __init__(
        self,
        runner: TerminalBenchRunner,
        task_id: str,
        max_turns: int = 16,
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
        # Seed the initial observation as a tool message before the first model turn
        # so the model immediately "sees" the terminal prompt/context.
        prompt = st.get("prompt")
        if isinstance(prompt, list):
            seeded_flag = st.get("_seeded_initial_obs", False)
            if not seeded_flag:
                self._append_tool_observation(prompt, initial_obs)
                st["prompt"] = prompt
                st["_seeded_initial_obs"] = True
        return st

    def _extract_command(self, messages: Messages) -> str:
        """Return the assistant's latest message content as a shell command.

        Prefers fenced bash/sh code blocks if present (joining lines with &&),
        otherwise falls back to the first non-empty line. Behavior unchanged; this
        documents the heuristic used by the adapter.
        """
        # For chat style, take last assistant content as the command.
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            content = str(messages)

        text = str(content)
        # Prefer a fenced bash/sh code block if present; join multiple lines with &&
        # Supported fences: ```bash|sh|shell ...``` or plain ``` ... ```
        fence_starts = ("```bash", "```sh", "```shell", "```")
        start_idx = -1
        start_len = 0
        for marker in fence_starts:
            idx = text.find(marker)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
                start_len = len(marker)
        if start_idx != -1:
            rest = text[start_idx + start_len :]
            end_idx = rest.find("```")
            if end_idx != -1:
                block = rest[:end_idx]
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if lines:
                    return " && ".join(lines)

        # Fallback: first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    def _append_tool_observation(self, messages: list[dict[str, Any]], obs: str) -> None:
        messages.append({"role": "tool", "content": obs, "name": "terminal"})

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:  # noqa: ARG002
        return bool(state.get("done") or state.get("turn", 0) >= self.max_turns)

    def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        st = self.setup_state(state)
        st["turn"] = st.get("turn", 0) + 1
        command = self._extract_command(messages)
        result: StepResult = self.runner.step(command)
        st.update({"done": result.done, "passed": result.passed, "observation": result.observation})
        # Clone messages to list for augmentation
        msgs: list[dict[str, Any]]
        if isinstance(messages, list):
            msgs = list(messages)
        else:
            msgs = [{"role": "user", "content": str(messages)}]
        self._append_tool_observation(msgs, result.observation)
        return msgs, st
