from __future__ import annotations

import json

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4SingleTurnEnv


def test_v4_single_turn_parallel_call_ordering():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": [""], "info": [{}]})
    env = BFCLV4SingleTurnEnv(dataset=ds)

    # Simulate two tool calls (we call call_tool sequentially to emulate ToolEnv processing ordering)
    msg1 = env.call_tool(
        "duckduckgo_search", json.dumps({"keywords": "bfcl", "max_results": 1, "region": "wt-wt"}), "t1"
    )
    msg2 = env.call_tool(
        "fetch_url_content", json.dumps({"url": "https://example", "mode": "truncate"}), "t2"
    )
    assert msg1["role"] == "tool" and msg1["tool_call_id"] == "t1"
    assert msg2["role"] == "tool" and msg2["tool_call_id"] == "t2"
