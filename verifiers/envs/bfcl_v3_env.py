from __future__ import annotations

import json
import ast
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
        info = it.get("info", {})
        if isinstance(info, list) and len(info) > 0:
            info = info[0]
        elif isinstance(info, list):
            info = {}
        info = dict(info)
        if "expected" in it:
            info["expected"] = it["expected"]
        # optional rubric specs for multi-turn
        if "final_state" in it:
            info["final_state"] = it["final_state"]
        if "tool_sequence" in it:
            info["tool_sequence"] = it["tool_sequence"]
        infos.append(info)
    return Dataset.from_dict({"prompt": prompts, "info": infos})


def _mk_v3_tools(statebox: dict[str, Any]) -> list[Callable]:
    def set_kv(key: str, value: str) -> str:
        statebox.setdefault("kv", {})[key] = value
        # record tool
        statebox.setdefault("tool_names", []).append("set_kv")
        return "OK"

    def get_kv(key: str) -> str:
        statebox.setdefault("tool_names", []).append("get_kv")
        return str(statebox.get("kv", {}).get(key, ""))

    def secret_add(x: int, y: int) -> int:
        statebox.setdefault("tool_names", []).append("secret_add")
        return x + y

    return [set_kv, get_kv, secret_add]


class BFCLV3Env(ToolEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        max_turns: int = 8,
        enable_missing_functions: bool = True,
        **kwargs: Any,
    ):
        self._statebox_ctor = self._create_statebox_factory
        self._statebox: dict[str, Any] = self._statebox_ctor()
        all_tools = _mk_v3_tools(self._statebox)
        self._withheld_name = "secret_add"
        self._withheld_tool = next(t for t in all_tools if t.__name__ == self._withheld_name)
        self._start_tools = [t for t in all_tools if t.__name__ != self._withheld_name] if enable_missing_functions else all_tools
        self._all_tools = all_tools
        self._enable_missing = enable_missing_functions
        self._revealed = False
        self._reveal_doc = V3Config().withheld_tool_doc
        super().__init__(
            tools=list(self._start_tools),  # Make a copy
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=Rubric(),
            **kwargs,
        )

    def _create_statebox_factory(self) -> dict[str, Any]:
        """Factory function to create a fresh statebox for each episode."""
        return {}

    def setup_state(self, state: State, **kwargs) -> State:
        # Reset state for new episode - prevent leakage between rollouts
        self._statebox = self._statebox_ctor()
        self._revealed = False
        
        # Reset tools to initial state (with or without withheld tool based on gating)
        self.tools = list(self._start_tools)  # Fresh copy
        self.tool_map = {t.__name__: t for t in self.tools}
        
        # Initialize state tracking for this episode
        state["tool_stage"] = 0
        state["kv"] = self._statebox.setdefault("kv", {})
        state["tool_names"] = self._statebox.setdefault("tool_names", [])
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
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        score_mode: Literal["exec", "ast"] = "exec",
        **kwargs: Any,
    ):
        self._statebox: dict[str, Any] = {}
        tools = _mk_v3_tools(self._statebox)
        super().__init__(
            tools=tools,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV3SingleTurnRubric(score_mode=score_mode),
            **kwargs,
        )
        # Force max_turns to 1 for single-turn environment
        self.max_turns = 1


# Minimal rubrics for v3

def _single_turn_exec_match(prompt: Messages, completion: Messages, answer: str, state: dict, info: dict, **kwargs) -> float:
    exp = (info or {}).get("expected", {})
    want = exp.get("output")
    if not want:
        return 0.0
    if isinstance(completion, list):
        for m in reversed(completion):
            if isinstance(m, dict) and m.get("role") == "tool":
                return 1.0 if str(m.get("content", "")) == str(want) else 0.0
    return 0.0


def _single_turn_ast_match(prompt: Messages, completion: Messages, answer: str, state: dict, info: dict, **kwargs) -> float:
    """AST-based single-turn rubric for leaderboard comparability with BFCL v1/v2."""
    exp = (info or {}).get("expected", {})
    want_tool = exp.get("tool")
    want_args = exp.get("args", {})
    
    if not want_tool:
        return 0.0
    
    if not isinstance(completion, list):
        return 0.0
    
    # Find the last tool call in completion
    tool_call = None
    for m in reversed(completion):
        if isinstance(m, dict) and m.get("role") == "assistant" and "tool_calls" in m:
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                tool_call = tool_calls[0]  # Take first tool call
                break
    
    if not tool_call:
        return 0.0
    
    # Parse tool call function and arguments
    try:
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            args_str = function.get("arguments", "{}")
            
            # Parse arguments
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str
        else:
            # Handle OpenAI function call format
            tool_name = tool_call.function.name if hasattr(tool_call, 'function') else None
            args = json.loads(tool_call.function.arguments) if hasattr(tool_call, 'function') else {}
        
        # Check tool name matches
        if tool_name != want_tool:
            return 0.0
        
        # Check arguments match (using AST equivalence for complex values)
        if not isinstance(want_args, dict) or not isinstance(args, dict):
            return 0.0
        
        # Check that all expected arguments are present with equivalent values
        for key, expected_value in want_args.items():
            if key not in args:
                return 0.0
            
            # For simple values, use direct comparison
            # For complex values, could use AST parsing for deeper equivalence
            if str(args[key]) != str(expected_value):
                return 0.0
        
        return 1.0
        
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        return 0.0


def _multiturn_state_goal(prompt: Messages, completion: Messages, answer: str, state: dict, info: dict, **kwargs) -> float:
    final = (info or {}).get("final_state", {})
    kv = state.get("kv", {})
    for k, v in final.items():
        if str(kv.get(k, None)) != str(v):
            return 0.0
    return 1.0 if final else 0.0


def _multiturn_sequence_match(prompt: Messages, completion: Messages, answer: str, state: dict, info: dict, **kwargs) -> float:
    want = (info or {}).get("tool_sequence", [])
    got = state.get("tool_names", [])
    if not want:
        return 0.0
    return 1.0 if list(want) == list(got[: len(want)]) else 0.0


class BFCLV3SingleTurnRubric(Rubric):
    def __init__(self, score_mode: Literal["exec", "ast"] = "exec"):
        if score_mode == "exec":
            funcs = [_single_turn_exec_match]
        elif score_mode == "ast":
            funcs = [_single_turn_ast_match]
        else:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
        super().__init__(funcs=funcs, weights=[1.0], parser=Parser())


class BFCLV3MultiTurnRubric(Rubric):
    def __init__(self):
        super().__init__(funcs=[_multiturn_state_goal, _multiturn_sequence_match], weights=[1.0, 0.5], parser=Parser())


def load_bfcl_v3(path: str | Path) -> Dataset:
    """Load BFCL v3 dataset from JSONL file."""
    items = _read_jsonl(path)
    return _to_dataset_v3(items)


def load_environment(
    version: Literal["v3"] = "v3",
    mode: Literal["multi", "single"] = "multi",
    dataset_file: str | None = None,
    dataset: Dataset | None = None,
    score_mode: Literal["exec", "ast"] = "exec",
    **kwargs: Any,
):
    ds = dataset
    if dataset_file:
        items = _read_jsonl(dataset_file)
        ds = _to_dataset_v3(items)
    if mode == "multi":
        return BFCLV3Env(dataset=ds, **kwargs)
    if mode == "single":
        return BFCLV3SingleTurnEnv(dataset=ds, score_mode=score_mode, **kwargs)
    raise ValueError(f"Unsupported v3 mode: {mode}")
