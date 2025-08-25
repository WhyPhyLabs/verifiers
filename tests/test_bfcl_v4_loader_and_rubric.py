from __future__ import annotations

import json

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4Rubric, load_environment as load_v4


def test_v4_loader_single_and_multi_constructs():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"], "info": [{}]})
    env_single = load_v4(version="v4", mode="single", single_turn_variant="b1", dataset=ds)
    env_multi = load_v4(version="v4", mode="multi", dataset=ds)
    assert env_single.max_turns == 1
    assert env_multi.max_turns > 1


def test_v4_rubric_class_scores_exact_match():
    rubric = BFCLV4Rubric()
    prompt = [{"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": json.dumps({"answer": "hello world", "context": "x"})}]
    gold = "Hello, World."
    score = rubric.get_reward_funcs()[0](parser=None, prompt=prompt, completion=completion, answer=gold, state={}, task="default", info={})
    assert score == 1.0
