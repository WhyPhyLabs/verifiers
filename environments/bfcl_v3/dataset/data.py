"""Dataset builder for BFCL v3 (Arrow-stable)."""

from __future__ import annotations
from typing import Iterable
from datasets import Dataset
import json

from .io import load_entries, load_possible_answers
from .prompt import build_prompt_messages, build_function_docs


def build_dataset(
    category: str,
    *,
    multi_turn_categories: Iterable[str],
    limit: int | None = None,
    seed: int | None = None,
) -> Dataset:
    entries = load_entries(category)
    answers = load_possible_answers(category)

    multi_turn_set = set(multi_turn_categories)

    records = []
    for entry in entries:
        messages_with_system = build_prompt_messages(
            entry, category=category, multi_turn_categories=multi_turn_set
        )
        # BFCL-specific fields encoded as JSON strings for Arrow stability
        function_docs = build_function_docs(
            entry, category=category, multi_turn_categories=multi_turn_set
        )
        # Remaining turns for multi-turn categories
        question_turns = entry.get("question", []) or []
        if category in multi_turn_set and question_turns:
            remaining_turns = [
                [{"role": m["role"], "content": m["content"]} for m in turn]
                for turn in question_turns[1:]
            ]
        else:
            remaining_turns = []
        initial_config = entry.get("initial_config", {}) or {}
        involved_classes = entry.get("involved_classes", []) or []
        language = (
            "Java"
            if category == "java"
            else "JavaScript"
            if category == "javascript"
            else "Python"
        )
        info = {
            "bfcl_category": category,
            "language": language,
            "bfcl_test_entry_id": entry.get("id"),
            # JSON-encoded fields for parity without runtime rehydration
            "function_json": json.dumps(function_docs),
            "initial_config_json": json.dumps(initial_config),
            "involved_classes_json": json.dumps(involved_classes),
            "bfcl_remaining_turns_json": json.dumps(remaining_turns),
        }
        records.append(
            {
                "id": entry["id"],
                "prompt": messages_with_system,
                # Store answer as JSON string to ensure a stable Arrow schema
                "answer": json.dumps(answers.get(entry["id"], [])),
                "task": category,
                "info": info,
            }
        )

    dataset = Dataset.from_list(records)

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


__all__ = [
    "build_dataset",
    "load_entries",
    "load_possible_answers",
]
