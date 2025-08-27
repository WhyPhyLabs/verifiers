from __future__ import annotations

import json

from datasets import Dataset

from verifiers import BinaryPassThroughRubric
from verifiers.envs.bfcl_v4_env import load_environment as load_v4


def test_v4_loader_single_and_multi_constructs():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"], "info": [{}]})
    env_single = load_v4(version="v4", mode="single", single_turn_variant="b1", dataset=ds)
    env_multi = load_v4(version="v4", mode="multi", dataset=ds)
    assert env_single.max_turns == 1
    assert env_multi.max_turns > 1


def test_v4_binary_scoring():
    """Test that BFCL v4 uses binary scoring via BinaryPassThroughRubric."""
    rubric = BinaryPassThroughRubric()
    
    # Test successful case - JSON answer matches after normalization
    prompt = [{"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": json.dumps({"answer": "hello world", "context": "x"})}]
    gold = "Hello, World."
    state = {"success": True}
    
    score = rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    )
    assert score.reward == 1.0
    assert score.metrics["binary_success"] == 1.0
    
    # Test failure case - JSON answer doesn't match
    state = {"success": False}
    score = rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    )
    assert score.reward == 0.0
    assert score.metrics["binary_success"] == 0.0


def test_v4_normalization():
    """Test that BFCL v4 normalization works correctly (used internally by env)."""
    from verifiers.envs.bfcl_v4_env import _normalize_answer
    
    # Test Unicode normalization
    assert _normalize_answer("Héllö, Wörld!") == "hello world"
    assert _normalize_answer("café") == "cafe"
    assert _normalize_answer("naïve") == "naive"
    
    # Test smart quotes and special Unicode characters
    assert _normalize_answer('"Hello" & "World"') == "hello world"
    assert _normalize_answer("'Hello' — 'World'") == "hello world"
    
    # Test whitespace normalization
    assert _normalize_answer("hello    world") == "hello world"
    assert _normalize_answer("  hello  world  ") == "hello world"
    assert _normalize_answer("hello\t\tworld\n") == "hello world"
    
    # Test punctuation removal
    assert _normalize_answer("Hello, World!") == "hello world"
    assert _normalize_answer("Hello.World") == "helloworld"
    assert _normalize_answer("Hello/World") == "helloworld"
    
    # Test case normalization
    assert _normalize_answer("HeLLo WoRLd") == "hello world"
    
    # Test mixed normalization
    assert _normalize_answer("  Héllö,   Wörld!  ") == "hello world"
    assert _normalize_answer('"CAFÉ" naïve...') == "cafe naive"


def test_v4_oracle_single_turn_env_has_max_turns_one():
    """Test that BFCLV4OracleSingleTurnEnv has max_turns=1 as required."""
    from verifiers.envs.bfcl_v4_env import BFCLV4OracleSingleTurnEnv
    
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    
    # Test with default constructor
    env = BFCLV4OracleSingleTurnEnv(dataset=ds)
    assert env.max_turns == 1, f"Expected max_turns=1, got {env.max_turns}"
    
    # Test with explicit parameters
    env2 = BFCLV4OracleSingleTurnEnv(dataset=ds, inject_oracle=True)
    assert env2.max_turns == 1, f"Expected max_turns=1, got {env2.max_turns}"
    
    # Test with inject_oracle=False
    env3 = BFCLV4OracleSingleTurnEnv(dataset=ds, inject_oracle=False)
    assert env3.max_turns == 1, f"Expected max_turns=1, got {env3.max_turns}"
