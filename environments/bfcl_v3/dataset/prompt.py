"""Build per-entry prompt messages for BFCL using bfcl_eval utilities."""

from __future__ import annotations

from typing import Any, Iterable

from bfcl_eval.model_handler.utils import (
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)

from .io import load_multi_turn_docs


def flatten_messages(
    question: Iterable[Iterable[dict[str, str]]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in question:
        for message in turn:
            messages.append({"role": message["role"], "content": message["content"]})
    return messages


def build_function_docs(
    entry: dict[str, Any], *, category: str, multi_turn_categories: Iterable[str]
) -> list[dict[str, Any]]:
    # Single-turn: docs come directly from the entry
    if category not in set(multi_turn_categories):
        return func_doc_language_specific_pre_processing(
            entry.get("function", []) or [], category
        )

    # Multi-turn: aggregate docs for involved classes
    docs: list[dict[str, Any]] = []
    for class_name in entry.get("involved_classes", []) or []:
        docs.extend(load_multi_turn_docs(class_name))
    return func_doc_language_specific_pre_processing(docs, category)


def build_prompt_messages(
    entry: dict[str, Any], *, category: str, multi_turn_categories: Iterable[str]
) -> list[dict[str, str]]:
    question_turns = entry.get("question", []) or []
    function_docs = build_function_docs(
        entry, category=category, multi_turn_categories=multi_turn_categories
    )

    if category in set(multi_turn_categories) and question_turns:
        first_turn_msgs = [
            {"role": m["role"], "content": m["content"]} for m in question_turns[0]
        ]
        return system_prompt_pre_processing_chat_model(
            first_turn_msgs, function_docs, category
        )

    # Single-turn or empty: include all user messages
    messages = flatten_messages(question_turns)
    return system_prompt_pre_processing_chat_model(messages, function_docs, category)


__all__ = [
    "flatten_messages",
    "build_function_docs",
    "build_prompt_messages",
]
