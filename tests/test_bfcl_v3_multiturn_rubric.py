from __future__ import annotations

from verifiers import BinaryPassThroughRubric


def test_v3_multiturn_binary_success():
    """Test that BFCL v3 multi-turn uses binary scoring via BinaryPassThroughRubric."""
    rubric = BinaryPassThroughRubric()
    
    # Test successful case - both state and sequence checks pass
    state = {"success": True, "kv": {"k": "v", "x": "1"}, "tool_names": ["set_kv", "secret_add"]}
    info = {"final_state": {"k": "v"}, "tool_sequence": ["set_kv", "secret_add"]}
    score = rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state,
        task="default",
        info=info
    )
    assert score.reward == 1.0
    assert score.metrics["binary_success"] == 1.0
    
    # Test failure case - either state or sequence check fails
    state = {"success": False, "kv": {"k": "wrong"}, "tool_names": ["set_kv"]}
    info = {"final_state": {"k": "v"}, "tool_sequence": ["set_kv", "secret_add"]}
    score = rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state,
        task="default",
        info=info
    )
    assert score.reward == 0.0
    assert score.metrics["binary_success"] == 0.0


def test_v3_multiturn_no_weights():
    """Test that v3 multi-turn no longer uses weights - it's binary only."""
    rubric = BinaryPassThroughRubric()
    
    # Verify that the rubric doesn't have reward_weights (binary scoring)
    assert not hasattr(rubric, 'reward_weights')
    
    # Test that scoring is purely binary based on success state
    state_success = {"success": True}
    state_failure = {"success": False}
    
    score_success = rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state_success,
        task="default",
        info={}
    )
    score_failure = rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state_failure,
        task="default",
        info={}
    )
    
    # Binary scoring: only 0.0 or 1.0, no partial credit
    assert score_success.reward == 1.0
    assert score_failure.reward == 0.0
