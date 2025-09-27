from __future__ import annotations

from typing import Any, Callable, Iterable

import verifiers as vf
from bfcl_eval.constants.category_mapping import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    TEST_FILE_MAPPING,
    VERSION_PREFIX,
)
from bfcl_eval.utils import (
    is_empty_output,
    is_executable_format_output,
    is_function_calling_format_output,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

from verifiers.parsers.parser import Parser
from verifiers.types import Info, RolloutScore
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from ..core.categories import (
    IRRELEVANCE_CATEGORIES,
    RELEVANCE_CATEGORIES,
    MULTI_TURN_CATEGORIES,
    AST_CATEGORIES,
)

__all__ = [
    "VERSION_PREFIX",
    "TEST_FILE_MAPPING",
    "MULTI_TURN_FUNC_DOC_FILE_MAPPING",
    "AST_CATEGORIES",
    "IRRELEVANCE_CATEGORIES",
    "RELEVANCE_CATEGORIES",
    "MULTI_TURN_CATEGORIES",
    "BFCLAstRubric",
    "BFCLRelevanceRubric",
    "BFCLIrrelevanceRubric",
    "BFCLMultiTurnRubric",
    "BFCLRubricGroup",
]


def _resolve_category(info: Info | None, task: str) -> str:
    if info:
        for key in ("bfcl_category", "category", "task"):
            value = info.get(key)  # type: ignore[arg-type]
            if isinstance(value, str) and value:
                return value
    return task


def _ensure_weights(
    funcs: Iterable[Callable[..., float]], weights: list[float] | None
) -> list[float]:
    if weights is not None:
        return weights
    return [1.0 for _ in funcs]


def _has_function_call(decoded: Any) -> bool:
    return is_function_calling_format_output(decoded)


def _has_execute_calls(decoded: Any) -> bool:
    return is_executable_format_output(decoded)


class _BFCLCategoryRubric(vf.Rubric):
    def __init__(
        self,
        *,
        categories: set[str],
        parser: Parser,
        funcs: list[Callable[..., float]] | None = None,
        weights: list[float] | None = None,
        parallelize_scoring: bool = True,
    ) -> None:
        funcs = funcs or []
        weights = _ensure_weights(funcs, weights)
        super().__init__(
            funcs=funcs,
            weights=weights,
            parser=parser,
            parallelize_scoring=parallelize_scoring,
        )
        self.categories = categories

    async def score_rollout(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: Any,
        state: vf.State,
        task: str = "default",
        info: Info | None = None,
        **kwargs: Any,
    ) -> RolloutScore:
        category = _resolve_category(info, task)
        if category not in self.categories:
            metrics = {name: 0.0 for name in self.get_reward_func_names()}
            return RolloutScore(reward=0.0, metrics=metrics)
        return await super().score_rollout(
            prompt,
            completion,
            answer,
            state,
            task,
            info,
            **kwargs,
        )


def bfcl_ast_structure_reward(
    parser: Parser,
    completion: vf.Messages,
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if _has_function_call(decoded) else 0.0


def bfcl_ast_exact_match_reward(
    parser: Parser,
    completion: vf.Messages,
    answer: Any,
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if decoded == answer else 0.0


def bfcl_execute_structure_reward(
    parser: Parser,
    completion: vf.Messages,
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if _has_execute_calls(decoded) else 0.0


def bfcl_execute_exact_match_reward(
    parser: Parser,
    completion: vf.Messages,
    answer: Any,
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if decoded == answer else 0.0


def bfcl_relevance_success_reward(
    parser: Parser,
    completion: vf.Messages,
    info: Info | None = None,
    task: str = "default",
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if _has_function_call(decoded) else 0.0


def bfcl_irrelevance_success_reward(
    parser: Parser,
    completion: vf.Messages,
    **_: Any,
) -> float:
    decoded = parser.parse_answer(completion)
    return 1.0 if is_empty_output(decoded) else 0.0


class BFCLAstRubric(_BFCLCategoryRubric):
    def __init__(
        self,
        *,
        categories: Iterable[str] = AST_CATEGORIES,
        parser: Parser,
        weights: list[float] | None = None,
        parallelize_scoring: bool = True,
        model_name: str = "gorilla-openfunctions-v2",
    ) -> None:
        format_reward = parser.get_format_reward_func()
        format_reward.__name__ = "bfcl_ast_format_reward"

        def bfcl_ast_semantic_reward(
            parser: Parser,
            completion: vf.Messages,
            answer: Any,
            info: Info | None = None,
            task: str = "default",
            **kwargs: Any,
        ) -> float:
            decoded = parser.parse_answer(completion)
            if decoded in (None, ""):
                return 0.0
            info = info or {}
            language = info.get("language") or (
                "Java"
                if task == "java"
                else "JavaScript"
                if task == "javascript"
                else "Python"
            )
            # Decode function docs from info JSON (no runtime rehydration)
            functions = []
            try:
                import json as _json

                functions = _json.loads(info.get("function_json", "[]"))
            except Exception:
                functions = []
            # Answers may be stored as JSON strings in the dataset
            try:
                import json as _json

                if isinstance(answer, str):
                    answer = _json.loads(answer)
            except Exception:
                pass
            try:
                result = ast_checker(
                    func_description=functions,
                    model_output=decoded,
                    possible_answer=answer,
                    language=language,
                    test_category=task,
                    model_name=model_name,
                )
                return 1.0 if result.get("valid", False) else 0.0
            except Exception:
                return 0.0

        bfcl_ast_semantic_reward.__name__ = "bfcl_ast_semantic_reward"

        funcs = [
            bfcl_ast_structure_reward,
            format_reward,
            bfcl_ast_semantic_reward,
        ]
        if weights is None:
            weights = [0.0, 0.0, 1.0]
        super().__init__(
            categories=set(categories),
            parser=parser,
            funcs=funcs,
            weights=weights,
            parallelize_scoring=parallelize_scoring,
        )


class BFCLRelevanceRubric(_BFCLCategoryRubric):
    def __init__(
        self,
        *,
        categories: Iterable[str] = RELEVANCE_CATEGORIES,
        parser: Parser,
        weights: list[float] | None = None,
        parallelize_scoring: bool = True,
    ) -> None:
        funcs = [
            bfcl_relevance_success_reward,
        ]
        super().__init__(
            categories=set(categories),
            parser=parser,
            funcs=funcs,
            weights=weights,
            parallelize_scoring=parallelize_scoring,
        )


class BFCLIrrelevanceRubric(_BFCLCategoryRubric):
    def __init__(
        self,
        *,
        categories: Iterable[str] = IRRELEVANCE_CATEGORIES,
        parser: Parser,
        weights: list[float] | None = None,
        parallelize_scoring: bool = True,
    ) -> None:
        funcs = [
            bfcl_irrelevance_success_reward,
        ]
        super().__init__(
            categories=set(categories),
            parser=parser,
            funcs=funcs,
            weights=weights,
            parallelize_scoring=parallelize_scoring,
        )


class BFCLMultiTurnRubric(_BFCLCategoryRubric):
    def __init__(
        self,
        *,
        categories: Iterable[str] = MULTI_TURN_CATEGORIES,
        parser: Parser,
        weights: list[float] | None = None,
        parallelize_scoring: bool = True,
        model_name: str = "gorilla-openfunctions-v2",
    ) -> None:
        # Full parity with BFCL's multi_turn_checker
        def bfcl_multi_turn_semantic_reward(
            parser: Parser,
            completion: vf.Messages,
            answer: Any,
            info: Info | None = None,
            task: str = "default",
            **_: Any,
        ) -> float:
            # Guard types
            if not isinstance(completion, list):
                return 0.0
            info = info or {}
            # Segment + decode via parser helper (keeps rubric parser-free)
            try:
                decode_multi = getattr(parser, "decode_multi_turn_execute")
            except Exception:
                return 0.0
            model_turns = decode_multi(completion)
            # Ground truth list[list[str]] from dataset; decode if json string
            gt_turns = answer or []
            try:
                import json as _json

                if isinstance(gt_turns, str):
                    gt_turns = _json.loads(gt_turns)
            except Exception:
                pass
            if not isinstance(gt_turns, list):
                return 0.0
            # Reconstruct minimal test_entry from info (no runtime rehydration)
            test_entry = {
                "id": info.get("bfcl_test_entry_id") or "multi_turn_unknown_0",
                "initial_config": {},
                "involved_classes": [],
            }
            try:
                import json as _json

                test_entry["initial_config"] = _json.loads(
                    info.get("initial_config_json", "{}")
                )
                test_entry["involved_classes"] = _json.loads(
                    info.get("involved_classes_json", "[]")
                )
            except Exception:
                pass
            try:
                result = multi_turn_checker(
                    multi_turn_model_result_list_decoded=model_turns,
                    multi_turn_ground_truth_list=gt_turns,
                    test_entry=test_entry,  # type: ignore[arg-type]
                    test_category=task,
                    model_name=model_name,
                )
                return 1.0 if bool(result.get("valid")) else 0.0
            except Exception:
                return 0.0

        bfcl_multi_turn_semantic_reward.__name__ = "bfcl_multi_turn_semantic_reward"

        # Keep structure/format as metrics but set their weights to 0 by default
        format_reward = parser.get_format_reward_func()
        format_reward.__name__ = "bfcl_multi_turn_format_reward"
        funcs = [
            bfcl_execute_structure_reward,  # metric only
            format_reward,  # metric only
            bfcl_multi_turn_semantic_reward,  # actual reward
        ]
        if weights is None:
            weights = [0.0, 0.0, 1.0]
        super().__init__(
            categories=set(categories),
            parser=parser,
            funcs=funcs,
            weights=weights,
            parallelize_scoring=parallelize_scoring,
        )


class BFCLRubricGroup(vf.RubricGroup):
    def __init__(
        self,
        *,
        ast_parser: Parser,
        execute_parser: Parser | None = None,
        include_relevance: bool = True,
        include_irrelevance: bool = True,
        include_multi_turn: bool = True,
        model_name: str = "gorilla-openfunctions-v2",
    ) -> None:
        rubrics: list[vf.Rubric] = [
            BFCLAstRubric(parser=ast_parser, model_name=model_name)
        ]

        if include_relevance and RELEVANCE_CATEGORIES:
            rubrics.append(BFCLRelevanceRubric(parser=ast_parser))

        if include_irrelevance and IRRELEVANCE_CATEGORIES:
            rubrics.append(BFCLIrrelevanceRubric(parser=ast_parser))

        if include_multi_turn and execute_parser is not None and MULTI_TURN_CATEGORIES:
            rubrics.append(
                BFCLMultiTurnRubric(parser=execute_parser, model_name=model_name)
            )

        super().__init__(rubrics=rubrics)
