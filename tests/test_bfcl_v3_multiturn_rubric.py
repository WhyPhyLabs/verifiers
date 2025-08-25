from __future__ import annotations

from verifiers.envs.bfcl_v3_env import BFCLV3MultiTurnRubric


def test_v3_multiturn_rubric_state_and_sequence():
    rubric = BFCLV3MultiTurnRubric()
    # State after rollout contains kv and tool call names in order
    state = {"kv": {"k": "v", "x": "1"}, "tool_names": ["set_kv", "secret_add"]}
    info = {"final_state": {"k": "v"}, "tool_sequence": ["set_kv", "secret_add"]}
    # Two funcs: state_goal and sequence_match
    funcs = rubric.get_reward_funcs()
    s_state = funcs[0](parser=None, prompt=[], completion=[], answer="", state=state, task="default", info=info)
    s_seq = funcs[1](parser=None, prompt=[], completion=[], answer="", state=state, task="default", info=info)
    assert s_state == 1.0 and s_seq == 1.0
