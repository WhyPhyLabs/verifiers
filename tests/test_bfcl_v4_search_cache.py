from __future__ import annotations

from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


def test_v4_search_cache_behavior(tmp_path):
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": [""], "info": [{}]})

    # First, populate cache by running without offline (cache dir None)
    env_online = BFCLV4WebEnv(dataset=ds, include_snippets=True, offline_cache_dir=None)
    # One result with snippet
    results = env_online.tool_map["duckduckgo_search"](keywords="bfcl v4", max_results=1, region="wt-wt")  # type: ignore
    assert isinstance(results, list) and len(results) == 1
    assert "title" in results[0] and "url" in results[0]
    assert "snippet" in results[0]

    # Now, create env with offline cache dir, but cache is empty → expect []
    env_offline_empty = BFCLV4WebEnv(dataset=ds, offline_cache_dir=str(tmp_path))
    empty = env_offline_empty.tool_map["duckduckgo_search"](keywords="missing", max_results=2, region="wt-wt")  # type: ignore
    assert empty == []

    # Populate cache for a known key using online env and write manually into cache of offline env
    # Use the same key formation logic: search::{region}::{max_results}::{keywords}
    # Note: The cache implementation sanitizes keys, so we need to use the sanitized version
    key = f"search__wt-wt__1__bfcl v4"  # sanitized version of "search::wt-wt::1::bfcl v4"
    # Get cached object from online env by calling again (ensures deterministic output)
    cached = env_online.tool_map["duckduckgo_search"](keywords="bfcl v4", max_results=1, region="wt-wt")  # type: ignore
    # Write to offline cache file
    (tmp_path / key).write_text(__import__("json").dumps(cached), encoding="utf-8")

    env_offline_hit = BFCLV4WebEnv(dataset=ds, offline_cache_dir=str(tmp_path))
    hit = env_offline_hit.tool_map["duckduckgo_search"](keywords="bfcl v4", max_results=1, region="wt-wt")  # type: ignore
    assert hit == cached
