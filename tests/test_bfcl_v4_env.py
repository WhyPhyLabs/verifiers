from __future__ import annotations

import json

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import (
    BFCLV4WebEnv,
    BFCLV4SingleTurnEnv,
    BFCLV4OracleSingleTurnEnv,
)


def test_v4_tools_failure_injection_deterministic(tmp_path):
    ds = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": "q"}]], "answer": [""], "info": [{}]}
    )
    env = BFCLV4WebEnv(
        dataset=ds,
        include_snippets=True,
        fetch_fail_rate=1.0,  # always fail
        failure_seed=42,
        offline_cache_dir=str(tmp_path),
        max_turns=2,
    )
    # Directly exercise tool call
    tool_msg = env.call_tool("fetch_url_content", json.dumps({"url": "https://x", "mode": "raw"}), "c1")
    assert tool_msg["role"] == "tool"
    assert isinstance(tool_msg["content"], str) and tool_msg["content"].startswith("ERROR:")


def test_v4_binary_scoring_normalization():
    """Test that BFCL v4 binary scoring handles normalization correctly."""
    from verifiers.envs.bfcl_v4_env import _normalize_answer
    
    # Test that normalization works correctly (this is used internally by the env)
    gold = "Hello, World."
    normalized_gold = _normalize_answer(gold)
    assert normalized_gold == "hello world"
    
    # Test that the environment uses binary scoring
    ds = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": "q"}]], "answer": [gold], "info": [{}]}
    )
    env = BFCLV4SingleTurnEnv(dataset=ds)
    
    # The environment should use BinaryPassThroughRubric
    assert hasattr(env, 'rubric')
    assert env.rubric.__class__.__name__ == 'BinaryPassThroughRubric'


def test_v4_single_turn_variants_construct():
    ds = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": "q"}]], "answer": [""], "info": [{}]}
    )
    env1 = BFCLV4SingleTurnEnv(dataset=ds, fetch_fail_rate=0.0)
    assert env1.max_turns == 1
    env2 = BFCLV4OracleSingleTurnEnv(dataset=ds)
    assert env2.max_turns == 1  # Oracle single-turn should be 1
