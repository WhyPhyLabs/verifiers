from __future__ import annotations

import pytest
from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


def test_v4_fetch_url_allowlist_blocks_non_http():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds)
    # Simulate blocked scheme
    out = env.tool_map["fetch_url_content"](url="file:///etc/passwd", mode="raw")  # type: ignore
    assert out.startswith("ERROR:")
