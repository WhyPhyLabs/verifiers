from __future__ import annotations

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


def test_v4_fetch_success_and_cache(tmp_path):
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": [""], "info": [{}]})

    # Online env writes cache on success
    env_online = BFCLV4WebEnv(dataset=ds, fetch_fail_rate=0.0, offline_cache_dir=None)
    body = env_online.tool_map["fetch_url_content"](url="https://example.com/page", mode="markdown")  # type: ignore
    assert isinstance(body, str) and body.startswith("# https://example.com/page")

    # Offline env with empty cache returns ""
    env_offline_empty = BFCLV4WebEnv(dataset=ds, offline_cache_dir=str(tmp_path))
    miss = env_offline_empty.tool_map["fetch_url_content"](url="https://absent", mode="raw")  # type: ignore
    assert miss == ""

    # Write a cache entry and then read it back
    key = f"page::raw::https://cached"
    (tmp_path / key).write_text("CACHED", encoding="utf-8")
    env_offline_hit = BFCLV4WebEnv(dataset=ds, offline_cache_dir=str(tmp_path))
    hit = env_offline_hit.tool_map["fetch_url_content"](url="https://cached", mode="raw")  # type: ignore
    assert hit == "CACHED"
