from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from datasets import Dataset

from verifiers import ToolEnv, SingleTurnEnv, Rubric, Parser
from verifiers.types import Messages, State


@dataclass
class V3Config:
    max_turns: int = 8
    withheld_tool_doc: str = (
        "New tool unlocked: secret_add(x: int, y: int) -> int. "
        "Use to add two integers when required."
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _to_dataset_v3(items: list[dict]) -> Dataset:
    prompts: list[list[dict]] = []
    infos: list[dict] = []
    for it in items:
        if "prompt" in it:
            prompts.append(it["prompt"])
        else:
            q = it.get("question") or it.get("query") or ""
            prompts.append([{"role": "user", "content": q}])
        info = dict(it.get("info", {}))
        if "expected" in it:
            info["expected"] = it["expected"]
        infos.append(info)
    return Dataset.from_dict({"prompt": prompts, "info": infos})


def _mk_v3_tools(statebox: dict[str, Any]) -> list[Callable]:
    def set_kv(key: str, value: str) -> str:
        """Set a string value in the environment state.

        Args:
            key (str): The key to set.
            value (str): The string value.
        """
        statebox.setdefault("kv", {})[key] = value
        return "OK"

    def get_kv(key: str) -> str:
        """Get a string value from the environment state.

        Args:
            key (str): The key to read.
        """
        return str(statebox.get("kv", {}).get(key, ""))

    def secret_add(x: int, y: int) -> int:
        """Add two integers (withheld initially in Missing-Functions scenarios).

        Args:
            x (int): first operand
            y (int): second operand
        """
        return x + y

    return [set_kv, get_kv, secret_add]


class BFCLV3Env(ToolEnv):
    """BFCL v3 multi-turn ToolEnv with missing-function gating."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        max_turns: int = 8,
        enable_missing_functions: bool = True,
        **kwargs: Any,
    ):
        self._statebox: dict[str, Any] = {}
        all_tools = _mk_v3_tools(self._statebox)
        self._withheld_name = "secret_add"
        self._withheld_tool = next(t for t in all_tools if t.__name__ == self._withheld_name)
        start_tools = [t for t in all_tools if t.__name__ != self._withheld_name] if enable_missing_functions else all_tools
        self._enable_missing = enable_missing_functions
        self._revealed = False
        self._reveal_doc = V3Config().withheld_tool_doc
        super().__init__(
            tools=start_tools,
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=Rubric(),
            **kwargs,
        )

    def setup_state(self, state: State, **kwargs) -> State:
        state["tool_stage"] = 0
        return state

    def _should_reveal_by_text(self, messages: Messages) -> bool:
        if not isinstance(messages, list) or not messages:
            return False
        last = messages[-1]
        if last.get("role") == "assistant" and isinstance(last.get("content"), str):
            txt = last["content"].lower()
            triggers = (
                "missing function",
                "reveal tool",
                "hidden tool",
                "unlock tool",
                "unavailable function",
            )
            return any(t in txt for t in triggers)
        return False

    def _should_reveal_by_attempt(self, messages: Messages) -> bool:
        if not isinstance(messages, list) or not messages:
            return False
        last = messages[-1]
        tcs = last.get("tool_calls") if isinstance(last, dict) else None
        if not tcs:
            return False
        for tc in tcs:
            try:
                name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
            except Exception:
                name = None
            if name == self._withheld_name:
                return True
        return False

    def _expose_withheld(self):
        if self._revealed:
            return
        self.tools.append(self._withheld_tool)
        self.tool_map[self._withheld_name] = self._withheld_tool
        self._revealed = True

    def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        extra: list[dict] = []
        if self._enable_missing and not self._revealed and self._should_reveal_by_attempt(messages):
            self._expose_withheld()
            extra.append({"role": "user", "content": self._reveal_doc})
            state["tool_stage"] = 1
            return extra, state

        tool_msgs: list[dict] = []
        if isinstance(messages, list) and messages and "tool_calls" in messages[-1]:
            tool_msgs, state = super().env_response(messages, state, **kwargs)

        if self._enable_missing and not self._revealed and self._should_reveal_by_text(messages):
            self._expose_withheld()
            extra.append({"role": "user", "content": self._reveal_doc})
            state["tool_stage"] = 1

        return tool_msgs + extra, state


class BFCLV3SingleTurnEnv(ToolEnv):
    """Single-turn 'next-action' probe built from v3 trajectories."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs: Any,
    ):
        self._statebox: dict[str, Any] = {}
        tools = _mk_v3_tools(self._statebox)
        super().__init__(
            tools=tools,
            max_turns=1,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=Rubric(),
            **kwargs,
        )


# Minimal rubrics for v3

def _single_turn_exec_match(prompt: Messages, completion: Messages, answer: str, state: dict, info: dict, **kwargs) -> float:
    """If info.expected.output is present, match last tool message content against it; else 0."""
    exp = (info or {}).get("expected", {})
    want = exp.get("output")
    if not want:
        return 0.0
    if isinstance(completion, list):
        # find last tool message
        for m in reversed(completion):
            if isinstance(m, dict) and m.get("role") == "tool":
                return 1.0 if str(m.get("content", "")) == str(want) else 0.0
    return 0.0


class BFCLV3SingleTurnRubric(Rubric):
    def __init__(self):
        super().__init__(funcs=[_single_turn_exec_match], weights=[1.0], parser=Parser())


def load_environment(
    version: Literal["v3"] = "v3",
    mode: Literal["multi", "single"] = "multi",
    dataset_file: str | None = None,
    dataset: Dataset | None = None,
    **kwargs: Any,
):
    ds = dataset
    if dataset_file:
        items = _read_jsonl(dataset_file)
        ds = _to_dataset_v3(items)
    if mode == "multi":
        return BFCLV3Env(dataset=ds, **kwargs)
    if mode == "single":
        return BFCLV3SingleTurnEnv(dataset=ds, **kwargs)
    raise ValueError(f"Unsupported v3 mode: {mode}")
