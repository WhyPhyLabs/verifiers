from __future__ import annotations

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4OracleSingleTurnEnv


def test_v4_oracle_injection_into_prompt():
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "answer the question"}]],
        "answer": ["foo"],
        "info": [{"evidence": "This is the gold evidence."}],
    })
    env = BFCLV4OracleSingleTurnEnv(dataset=ds)
    injected = env.dataset[0]["prompt"][0]["content"]
    assert "gold evidence" in injected.lower()
