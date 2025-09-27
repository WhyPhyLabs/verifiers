from __future__ import annotations

from typing import Iterable
from datasets import Dataset

import verifiers as vf

from ..envs.environments import (
    MULTI_TURN_CATEGORIES,
    BFCLSingleTurnToolEnv,
    BFCLMultiTurnStatefulToolEnv,
    BFCLCategoryEnv,
)
from ..core.categories import CATEGORY_NAMES


def _validate_categories(categories: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for category in categories:
        if category not in CATEGORY_NAMES:
            raise ValueError(
                f"Unknown BFCL category '{category}'. Available: {sorted(CATEGORY_NAMES)}"
            )
        normalized.append(category)
    return normalized


def load_environment(
    *,
    categories: Iterable[str] | None = None,
    limit: int | None = None,
    seed: int | None = None,
    use_tools: bool = False,
    datasets: dict[str, Dataset] | None = None,
    eval_datasets: dict[str, Dataset] | None = None,
) -> vf.Environment:
    """Create an EnvGroup for the requested BFCL v3 categories."""

    selected = _validate_categories(categories or CATEGORY_NAMES)

    datasets = datasets or {}
    eval_datasets = eval_datasets or {}

    env_instances: list[vf.Environment] = []
    for category in selected:
        if use_tools:
            if category in MULTI_TURN_CATEGORIES:
                env_instances.append(
                    BFCLMultiTurnStatefulToolEnv(
                        limit=limit,
                        seed=seed,
                        category=category,
                        dataset=datasets.get(category),
                        eval_dataset=eval_datasets.get(category),
                    )
                )
            else:
                env_instances.append(
                    BFCLSingleTurnToolEnv(
                        limit=limit,
                        seed=seed,
                        category=category,
                        dataset=datasets.get(category),
                        eval_dataset=eval_datasets.get(category),
                    )
                )
        else:
            env_instances.append(
                BFCLCategoryEnv(
                    limit=limit,
                    seed=seed,
                    category=category,
                    dataset=datasets.get(category),
                    eval_dataset=eval_datasets.get(category),
                )
            )

    return vf.EnvGroup(env_instances, env_names=selected)
