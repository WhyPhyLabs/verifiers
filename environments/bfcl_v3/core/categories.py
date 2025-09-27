from __future__ import annotations

from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING

IRRELEVANCE_CATEGORIES: set[str] = {
    c for c in TEST_COLLECTION_MAPPING["ast"] if "irrelevance" in c
}
RELEVANCE_CATEGORIES: set[str] = {
    c
    for c in TEST_COLLECTION_MAPPING["ast"]
    if "relevance" in c and "irrelevance" not in c
}
MULTI_TURN_CATEGORIES: set[str] = set(TEST_COLLECTION_MAPPING["multi_turn"])
AST_CATEGORIES: set[str] = (
    set(TEST_COLLECTION_MAPPING["ast"]) - IRRELEVANCE_CATEGORIES - RELEVANCE_CATEGORIES
)

CATEGORY_NAMES = list(TEST_COLLECTION_MAPPING["all"])  # canonical order
