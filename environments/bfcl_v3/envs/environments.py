"""BFCL v3 environment classes (baseline, tool, and stateful-tool)."""

from __future__ import annotations

import verifiers as vf
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    CLASS_FILE_PATH_MAPPING as BFCL_CLASS_MODULES,
)


from typing import Any, Callable
import importlib
import inspect
import json

from ..dataset.data import build_dataset
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.model_style import ModelStyle
from ..parsers.parser import BFCLParser
from ..rubrics.rubric import BFCLRubricGroup


from ..core.categories import (
    IRRELEVANCE_CATEGORIES,
    RELEVANCE_CATEGORIES,
    MULTI_TURN_CATEGORIES,
    AST_CATEGORIES,
)


class BFCLCategoryEnv(vf.SingleTurnEnv):
    """Single BFCL category environment (baseline, no tools)."""

    def __init__(
        self,
        category: str,
        *,
        ast_parser: vf.Parser | None = None,
        execute_parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        limit: int | None = None,
        seed: int | None = None,
        dataset=None,
        eval_dataset=None,
    ) -> None:
        self.category = category
        # Choose language for AST decoding based on category
        lang = (
            "Java"
            if category == "java"
            else "JavaScript"
            if category == "javascript"
            else "Python"
        )
        ast_parser = ast_parser or BFCLParser(language=lang)

        include_multi_turn = category in MULTI_TURN_CATEGORIES
        if include_multi_turn:
            execute_parser = execute_parser or BFCLParser(mode="execute")

        if rubric is None:
            rubric = BFCLRubricGroup(
                ast_parser=ast_parser,
                execute_parser=execute_parser if include_multi_turn else None,
                include_relevance=category in RELEVANCE_CATEGORIES,
                include_irrelevance=category in IRRELEVANCE_CATEGORIES,
                include_multi_turn=include_multi_turn,
            )

        if dataset is None:
            dataset = build_dataset(
                category,
                multi_turn_categories=MULTI_TURN_CATEGORIES,
                limit=limit,
                seed=seed,
            )
        if eval_dataset is None:
            eval_dataset = dataset

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=ast_parser,
            rubric=rubric,
        )

        self.category_name = category


def make_base_env(
    category: str, *, limit: int | None = None, seed: int | None = None
) -> BFCLCategoryEnv:
    return BFCLCategoryEnv(category=category, limit=limit, seed=seed)


CATEGORY_NAMES = list(TEST_COLLECTION_MAPPING["all"])  # canonical order


__all__ = [
    "BFCLCategoryEnv",
    "make_base_env",
    "BFCLSingleTurnToolEnv",
    "BFCLMultiTurnStatefulToolEnv",
    "CATEGORY_NAMES",
    "IRRELEVANCE_CATEGORIES",
    "RELEVANCE_CATEGORIES",
    "MULTI_TURN_CATEGORIES",
    "AST_CATEGORIES",
]


def _build_stub_tool_map(
    function_docs: list[dict[str, Any]],
) -> dict[str, Callable[..., str]]:
    """Create no-op tool callables matching BFCL function names.

    Each callable returns a short acknowledgement string; rubric checks do not
    depend on tool outputs, only on assistant messages and decoded content.
    """

    def make_stub(name: str) -> Callable[..., str]:
        def _stub(**kwargs: Any) -> str:
            return f"{name} executed"

        _stub.__name__ = name
        return _stub

    tools: dict[str, Callable[..., str]] = {}
    for f in function_docs or []:
        name = f.get("name")
        if isinstance(name, str) and name:
            tools[name] = make_stub(name)
    return tools


class BFCLSingleTurnToolEnv(vf.ToolEnv):
    """Single-turn BFCL category with tool calling enabled.

    - Provides per-sample tool schemas via dataset `info.oai_tools`.
    - Uses stub Python callables so the loop can return tool messages.
    """

    def __init__(
        self,
        *,
        category: str,
        ast_parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        limit: int | None = None,
        seed: int | None = None,
        max_turns: int = 6,
        dataset=None,
        eval_dataset=None,
    ) -> None:
        self.category = category
        if category in MULTI_TURN_CATEGORIES:
            raise ValueError(
                "Use BFCLMultiTurnStatefulToolEnv for multi-turn categories"
            )

        lang = (
            "Java"
            if category == "java"
            else "JavaScript"
            if category == "javascript"
            else "Python"
        )
        ast_parser = ast_parser or BFCLParser(language=lang)

        rubric = rubric or BFCLRubricGroup(
            ast_parser=ast_parser,
            include_relevance=category in RELEVANCE_CATEGORIES,
            include_irrelevance=category in IRRELEVANCE_CATEGORIES,
            include_multi_turn=False,
        )

        if dataset is None:
            dataset = build_dataset(
                category,
                multi_turn_categories=MULTI_TURN_CATEGORIES,
                limit=limit,
                seed=seed,
            )
        if eval_dataset is None:
            eval_dataset = dataset

        # Initialize without tools; set env-level oai_tools and stub map below
        super().__init__(
            tools=[],
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=ast_parser,
            rubric=rubric,
        )

        # Category-wide BFCL tool schemas so first turn uses them
        # Build env-level tool schema superset from dataset info
        try:
            import json as _json

            docs: list[dict] = []
            for row in self.get_dataset(-1):
                info = row.get("info", {})
                if isinstance(info, dict):
                    try:
                        docs.extend(_json.loads(info.get("function_json", "[]")))
                    except Exception:
                        pass
            self.oai_tools = convert_to_tool(
                docs, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Responses
            )
        except Exception:
            self.oai_tools = []
        # Build stub tool_map keyed by OAI tool names
        stub_map: dict[str, Callable[..., str]] = {}
        for t in self.oai_tools:
            name = (
                t.get("function", {}).get("name") if "function" in t else t.get("name")
            )
            if isinstance(name, str) and name:
                stub_map[name] = _build_stub_tool_map([{"name": name}])[name]
        self.tool_map.update(stub_map)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:  # type: ignore[override]
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> tuple[vf.Messages, vf.State]:  # type: ignore[override]
        """Support dict-style tool_calls in addition to SDK objects.

        Mirrors BFCLMultiTurnStatefulToolEnv handling so OpenAI REST responses
        (which use plain dicts) work without crashes.
        """
        assert isinstance(messages, list)
        assert "tool_calls" in messages[-1]
        tool_messages: list[vf.Message] = []
        for tool_call in messages[-1]["tool_calls"]:
            # Extract name/args/id for dicts or SDK objects
            if isinstance(tool_call, dict):
                func = tool_call.get("function") or {}
                tool_name = func.get("name", "") or ""
                args_obj = func.get("arguments", {})
                if isinstance(args_obj, str):
                    try:
                        tool_args = json.loads(args_obj)
                    except Exception:
                        tool_args = {}
                elif isinstance(args_obj, dict):
                    tool_args = args_obj
                else:
                    tool_args = {}
                tool_call_id = tool_call.get("id", "") or ""
            else:
                tool_name = getattr(
                    getattr(tool_call, "function", object()), "name", ""
                )
                args_str = getattr(
                    getattr(tool_call, "function", object()), "arguments", None
                )
                try:
                    tool_args = json.loads(args_str or "{}")
                except Exception:
                    tool_args = {}
                tool_call_id = getattr(tool_call, "id", "") or ""

            # Call mapped tool if available; otherwise return an error string
            if tool_name in getattr(self, "tool_map", {}):
                tool_message: vf.Message = await self.call_tool(
                    tool_name, tool_args, tool_call_id
                )
            else:
                tool_message = {
                    "role": "tool",
                    "content": f"Tool '{tool_name}' is not available.",
                    "tool_call_id": tool_call_id,
                }
            tool_messages.append(tool_message)
        return tool_messages, state


class BFCLMultiTurnStatefulToolEnv(vf.StatefulToolEnv):
    """Multi-turn BFCL category with stateful tools and user-turn interleaving."""

    category: str

    def __init__(
        self,
        *,
        category: str,
        ast_parser: vf.Parser | None = None,
        execute_parser: vf.Parser | None = None,
        rubric: vf.Rubric | None = None,
        limit: int | None = None,
        seed: int | None = None,
        max_turns: int = 20,
        dataset=None,
        eval_dataset=None,
    ) -> None:
        self.category = category
        if category not in MULTI_TURN_CATEGORIES:
            raise ValueError(
                "BFCLMultiTurnStatefulToolEnv is only for multi-turn categories"
            )

        lang = "Python"  # execution decoding is language-agnostic but keep for parity
        ast_parser = ast_parser or BFCLParser(language=lang)
        execute_parser = execute_parser or BFCLParser(mode="execute")
        rubric = rubric or BFCLRubricGroup(
            ast_parser=ast_parser,
            execute_parser=execute_parser,
            include_relevance=False,
            include_irrelevance=False,
            include_multi_turn=True,
        )

        if dataset is None:
            dataset = build_dataset(
                category,
                multi_turn_categories=MULTI_TURN_CATEGORIES,
                limit=limit,
                seed=seed,
            )
        if eval_dataset is None:
            eval_dataset = dataset

        # tools are per-sample; initialize with none at env-level
        super().__init__(
            tools=[],
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=execute_parser,
            rubric=rubric,
        )
        # Provide env-level tool schemas built from dataset info for first turn
        try:
            import json as _json

            docs: list[dict] = []
            for row in self.get_dataset(-1):
                info = row.get("info", {})
                if isinstance(info, dict):
                    try:
                        docs.extend(_json.loads(info.get("function_json", "[]")))
                    except Exception:
                        pass
            self.oai_tools = convert_to_tool(
                docs, GORILLA_TO_OPENAPI, ModelStyle.OpenAI_Responses
            )
        except Exception:
            self.oai_tools = []

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:  # type: ignore[override]
        info = state.get("info", {})
        cat = info.get("bfcl_category", "")
        import json as _json

        try:
            involved = _json.loads(info.get("involved_classes_json", "[]"))
        except Exception:
            involved = []
        try:
            init_conf = _json.loads(info.get("initial_config_json", "{}"))
        except Exception:
            init_conf = {}
        try:
            remaining_turns = _json.loads(info.get("bfcl_remaining_turns_json", "[]"))
        except Exception:
            remaining_turns = []
        long_context = "long_context" in str(cat)

        instances: dict[str, Any] = {}
        method_to_instance: dict[str, Any] = {}

        # Instantiate BFCL classes and pre-load scenarios
        for class_name in involved:
            module_path = BFCL_CLASS_MODULES.get(class_name)
            if not module_path:
                continue
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            inst = cls()
            if hasattr(inst, "_load_scenario"):
                try:
                    inst._load_scenario(
                        init_conf.get(class_name, {}), long_context=long_context
                    )  # type: ignore[arg-type]
                except Exception:
                    # Fallback without long_context if signature differs
                    try:
                        inst._load_scenario(init_conf.get(class_name, {}))  # type: ignore[arg-type]
                    except Exception:
                        pass
            # Collect public methods
            for name, method in inspect.getmembers(inst, predicate=inspect.ismethod):
                if name.startswith("_"):
                    continue
                method_to_instance[name] = inst
            instances[class_name] = inst

        # Build tool map wrappers that dispatch to instance methods
        tool_map: dict[str, Callable[..., Any]] = {}

        def make_caller(mname: str, inst: Any) -> Callable[..., Any]:
            def _call(**kwargs: Any) -> Any:
                fn = getattr(inst, mname)
                return fn(**kwargs)

            _call.__name__ = mname
            return _call

        for mname, inst in method_to_instance.items():
            tool_map[mname] = make_caller(mname, inst)

        state["bfcl_instances"] = instances
        state["bfcl_method_to_instance"] = method_to_instance
        state["tool_map"] = tool_map
        state["bfcl_remaining_turns"] = remaining_turns
        return state

    async def is_completed(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> bool:  # type: ignore[override]
        assert isinstance(messages, list)
        if not messages:
            return False
        last = messages[-1]
        # Continue if assistant emitted tool calls
        if last.get("role") == "assistant" and last.get("tool_calls"):
            return False
        # Stop only when assistant message with no tool_calls and no remaining user turns
        remaining = state.get("bfcl_remaining_turns", [])
        if (
            last.get("role") == "assistant"
            and not last.get("tool_calls")
            and not remaining
        ):
            return True
        return False

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> tuple[vf.Messages, vf.State]:  # type: ignore[override]
        assert isinstance(messages, list)
        last = messages[-1]
        # If the assistant made tool calls, dispatch them
        if last.get("role") == "assistant" and last.get("tool_calls"):
            # Support both OpenAI SDK objects and plain dicts for tool calls.
            tool_messages: list[vf.Message] = []
            for tool_call in last["tool_calls"]:
                # Extract name/args/id from either mapping or SDK object
                if isinstance(tool_call, dict):
                    func = tool_call.get("function") or {}
                    tool_name = func.get("name", "") or ""
                    args_obj = func.get("arguments", {})
                    if isinstance(args_obj, str):
                        try:
                            tool_args = json.loads(args_obj)
                        except Exception:
                            tool_args = {}
                    elif isinstance(args_obj, dict):
                        tool_args = args_obj
                    else:
                        tool_args = {}
                    tool_call_id = tool_call.get("id", "") or ""
                else:
                    # Fallback: typed ChatCompletionMessageToolCall
                    tool_name = getattr(
                        getattr(tool_call, "function", object()), "name", ""
                    )
                    args_str = getattr(
                        getattr(tool_call, "function", object()), "arguments", None
                    )
                    try:
                        tool_args = json.loads(args_str or "{}")
                    except Exception:
                        tool_args = {}
                    tool_call_id = getattr(tool_call, "id", "") or ""

                fn = state.get("tool_map", {}).get(tool_name)
                if fn is None:
                    content = f"Tool '{tool_name}' is not available."
                else:
                    try:
                        result = fn(**tool_args)
                        # normalize complex objects to string
                        content = (
                            result
                            if isinstance(result, str)
                            else json.dumps(result, default=str)
                        )
                    except Exception as e:  # pragma: no cover
                        content = f"Error: {e}"
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id,
                    }
                )
            return tool_messages, state

        # Otherwise, interleave the next BFCL user turn if available
        remaining: list[list[vf.Message]] = state.get("bfcl_remaining_turns", [])
        if remaining:
            next_turn = remaining.pop(0)
            state["bfcl_remaining_turns"] = remaining
            return next_turn, state

        return [], state

    def update_tool_args(
        self, tool_args: dict, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> dict:  # type: ignore[override]
        # BFCL APIs do not require per-call arg rewriting; return as-is.
        return tool_args
