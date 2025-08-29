



from __future__ import annotations

import json
import asyncio

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
    
    score = asyncio.run(rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 1.0
    assert score.metrics["binary_success"] == 1.0
    
    # Test failure case - JSON answer doesn't match
    state = {"success": False}
    score = asyncio.run(rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 0.0
    assert score.metrics["binary_success"] == 0.0


def test_v4_normalization():
    """Test that v4 normalization matches BFCL spec exactly."""
    rubric = BinaryPassThroughRubric()
    
    # Test case-insensitive and punctuation-insensitive matching
    prompt = [{"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": json.dumps({"answer": "Hello, World!"})}]
    gold = "hello world"
    state = {"success": True}
    
    score = asyncio.run(rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 1.0
    assert score.metrics["binary_success"] == 1.0
    
    # Test that diacritics are preserved (not normalized away)
    completion = [{"role": "assistant", "content": json.dumps({"answer": "résumé"})}]
    gold = "resume"
    state = {"success": False}  # Should fail because diacritics are preserved
    
    score = asyncio.run(rubric.score_rollout(
        prompt=prompt,
        completion=completion,
        answer=gold,
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 0.0
    assert score.metrics["binary_success"] == 0.0


def test_v4_normalization_function():
    """Test the _normalize_answer function directly."""
    from verifiers.envs.bfcl_v4_env import _normalize_answer
    
    # Test BFCL-specified punctuation removal
    assert _normalize_answer("Hello, World.") == "hello world"     # comma and period
    assert _normalize_answer("Hello/World") == "helloworld"        # slash
    assert _normalize_answer("Hello-World") == "helloworld"       # dash
    assert _normalize_answer("Hello_World") == "helloworld"       # underscore
    assert _normalize_answer("Hello*World") == "helloworld"       # asterisk
    assert _normalize_answer("Hello^World") == "helloworld"       # caret
    assert _normalize_answer("Hello(World)") == "helloworld"     # parentheses
    assert _normalize_answer('"Hello"') == "hello"               # quotes
    assert _normalize_answer("'Hello'") == "hello"               # single quotes
    
    # Test that non-BFCL punctuation is preserved
    assert _normalize_answer("Hello!World") == "hello!world"     # preserves exclamation (not in BFCL set)
    assert _normalize_answer("Hello=World") == "hello=world"     # preserves equals (not in BFCL set)
    
    # Test that whitespace is NOT collapsed (BFCL-exact doesn't do this)
    assert _normalize_answer("hello    world") == "hello    world"  # preserves extra spaces
    assert _normalize_answer("  hello  world  ") == "  hello  world  "  # preserves leading/trailing spaces
    
    # Test case normalization
    assert _normalize_answer("HeLLo WoRLd") == "hello world"
    
    # Test Unicode characters are preserved (not normalized to ASCII)
    assert _normalize_answer("Ωmega") == "Ωmega"  # preserves Unicode symbol
    assert _normalize_answer("café") == "café"     # preserves accented characters



