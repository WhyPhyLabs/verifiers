
from __future__ import annotations

import asyncio
from verifiers import BinaryPassThroughRubric


def test_v3_single_turn_binary_success():
    """Test that BFCL v3 single-turn uses binary scoring via BinaryPassThroughRubric."""
    rubric = BinaryPassThroughRubric()
    
    # Test successful case
    state = {"success": True}
    score = asyncio.run(rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 1.0
    assert score.metrics["binary_success"] == 1.0
    
    # Test failure case
    state = {"success": False}
    score = asyncio.run(rubric.score_rollout(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "test"}],
        answer="test",
        state=state,
        task="default",
        info={}
    ))
    assert score.reward == 0.0
    assert score.metrics["binary_success"] == 0.0
