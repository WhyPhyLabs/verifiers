"""Read packaged BFCL datasets via bfcl_eval + importlib.resources."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

from bfcl_eval.constants.category_mapping import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    TEST_FILE_MAPPING,
)
from bfcl_eval.utils import load_file


def load_possible_answers(category: str) -> dict[str, Any]:
    filename = TEST_FILE_MAPPING[category]
    resource = resources.files("bfcl_eval").joinpath(
        "data", "possible_answer", filename
    )
    if not resource.exists():
        return {}
    with resources.as_file(resource) as path:
        return {row["id"]: row["ground_truth"] for row in load_file(path)}


@lru_cache(maxsize=None)
def load_entries(category: str) -> list[dict[str, Any]]:
    filename = TEST_FILE_MAPPING[category]
    resource = resources.files("bfcl_eval").joinpath("data", filename)
    with resources.as_file(resource) as path:
        return load_file(path)


@lru_cache(maxsize=None)
def load_multi_turn_docs(class_name: str) -> list[dict[str, Any]]:
    filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
    if filename is None:
        return []
    resource = resources.files("bfcl_eval").joinpath(
        "data", "multi_turn_func_doc", filename
    )
    if not resource.exists():
        return []
    with resources.as_file(resource) as path:
        return load_file(path)


def get_entry_by_id(category: str, entry_id: str) -> dict[str, Any] | None:
    for entry in load_entries(category):
        if entry.get("id") == entry_id:
            return entry
    return None


__all__ = [
    "load_possible_answers",
    "load_entries",
    "load_multi_turn_docs",
    "get_entry_by_id",
]
