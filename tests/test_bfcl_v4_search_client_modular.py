from __future__ import annotations

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


class StubSearchClient:
    def search(self, keywords: str, max_results: int = 10, region: str = "wt-wt"):
        return [{"title": "stub", "url": "https://example.com", "snippet": "s"}][:max_results]


def test_v4_modular_search_client_injection():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds, search_client=StubSearchClient())
    res = env.tool_map["duckduckgo_search"](keywords="x", max_results=1, region="wt-wt")  # type: ignore
    assert res and res[0]["title"] == "stub"
