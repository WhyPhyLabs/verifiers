from __future__ import annotations

from verifiers.envs.bfcl_v3_env import BFCLV3SingleTurnRubric


def test_v3_single_turn_rubric_matches_output_when_present():
    rubric = BFCLV3SingleTurnRubric()
    info = {"expected": {"tool": "set_kv", "args": {"key": "k", "value": "v"}, "output": "OK"}}
    completion = [
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "OK", "tool_call_id": "t1"},
    ]
    score = rubric.get_reward_funcs()[0](parser=None, prompt=[], completion=completion, answer="", state={}, task="default", info=info)
    assert score == 1.0
