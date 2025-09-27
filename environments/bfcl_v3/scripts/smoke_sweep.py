"""Quick local smoke sweep for BFCL v3 environments.

Usage:
  uv run python -m environments.bfcl_v3.scripts.smoke_sweep --limit 1 --use-tools

This runs one sample per category (by default), injects dict-style tool calls
when tools are available, and scores with BFCLRubricGroup. Intended for devs.
"""

from __future__ import annotations
import argparse
import asyncio
import json
from typing import Iterable

import verifiers as vf

from environments.bfcl_v3.core.categories import (
    CATEGORY_NAMES,
    MULTI_TURN_CATEGORIES,
)
from environments.bfcl_v3.dataset.data import build_dataset
from environments.bfcl_v3.envs.environments import (
    BFCLMultiTurnStatefulToolEnv,
    BFCLSingleTurnToolEnv,
)
from environments.bfcl_v3.parsers.parser import BFCLParser
from environments.bfcl_v3.rubrics.rubric import BFCLRubricGroup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BFCL v3 smoke sweep")
    p.add_argument(
        "--categories",
        nargs="*",
        default=CATEGORY_NAMES,
        help="Subset of categories (default: all)",
    )
    p.add_argument("--limit", type=int, default=1, help="Samples per category")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    p.add_argument(
        "--use-tools",
        action="store_true",
        help="Enable tool-mode envs for both single and multi-turn",
    )
    return p.parse_args()


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _inject_dict_tool_call_single(env: BFCLSingleTurnToolEnv, row) -> list[vf.Message]:
    # Use any available tool name; otherwise send empty assistant content
    tool_name = next(iter(getattr(env, "tool_map", {}) or {}), None)
    messages = list(row["prompt"])  # copy
    if tool_name:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps({})},
                    }
                ],
            }
        )
    else:
        messages.append({"role": "assistant", "content": "[]"})
    return messages


def _inject_dict_tool_call_multi(
    env: BFCLMultiTurnStatefulToolEnv, row, state
) -> list[vf.Message]:
    tool_name = next(iter(state.get("tool_map", {}) or {}), None)
    messages = list(row["prompt"])  # copy
    if tool_name:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps({})},
                    }
                ],
            }
        )
    else:
        messages.append({"role": "assistant", "content": "[]"})
    return messages


def main() -> None:
    args = parse_args()
    selected: Iterable[str] = args.categories
    results: list[tuple[str, float]] = []

    for cat in selected:
        ds = build_dataset(
            cat,
            multi_turn_categories=MULTI_TURN_CATEGORIES,
            limit=args.limit,
            seed=args.seed,
        )
        if len(ds) == 0:
            print(f"- {cat:25s} SKIP (no data)")
            continue
        row = ds[0]
        if cat in MULTI_TURN_CATEGORIES:
            env = BFCLMultiTurnStatefulToolEnv(
                category=cat, dataset=ds, eval_dataset=ds
            )
            state: vf.State = {"timing": {"total_ms": 0}, "info": row["info"]}
            state = asyncio.get_event_loop().run_until_complete(env.setup_state(state))
            messages = _inject_dict_tool_call_multi(env, row, state)
            if messages[-1].get("tool_calls"):
                tool_msgs, state = asyncio.get_event_loop().run_until_complete(
                    env.env_response(messages, state)
                )
                messages += tool_msgs
            rubric = BFCLRubricGroup(
                ast_parser=BFCLParser(language="Python"),
                execute_parser=BFCLParser(mode="execute"),
                include_relevance=False,
                include_irrelevance=False,
                include_multi_turn=True,
            )
        else:
            if args.use_tools:
                env = BFCLSingleTurnToolEnv(category=cat, dataset=ds, eval_dataset=ds)
                state: vf.State = {"timing": {"total_ms": 0}, "info": row["info"]}
                state = asyncio.get_event_loop().run_until_complete(
                    env.setup_state(state)
                )
                messages = _inject_dict_tool_call_single(env, row)
                if messages[-1].get("tool_calls"):
                    tool_msgs, state = asyncio.get_event_loop().run_until_complete(
                        env.env_response(messages, state)
                    )
                    messages += tool_msgs
            else:
                env = None
                messages = list(row["prompt"]) + [
                    {"role": "assistant", "content": "[]"}
                ]
            rubric = BFCLRubricGroup(
                ast_parser=BFCLParser(language="Python"),
                include_relevance=("relevance" in cat),
                include_irrelevance=("irrelevance" in cat),
                include_multi_turn=False,
            )

        score = asyncio.get_event_loop().run_until_complete(
            rubric.score_rollout(
                row["prompt"],
                messages,
                row["answer"],
                state if "state" in locals() else {"timing": {"total_ms": 0}},
                task=cat,
                info=row["info"],
            )
        )
        print(
            f"- {cat:25s} reward={score.reward:.2f} metrics={list(score.metrics.keys())[:3]}..."
        )
        results.append((cat, score.reward))

    print(f"Sweep complete: {len(results)} categories")


if __name__ == "__main__":
    main()
