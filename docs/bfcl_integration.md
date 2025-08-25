BFCL Integration (v3 and v4)

- v3 (multi-turn): BFCLV3Env (ToolEnv)
  - Stateful tools; “Missing-Functions” gating reveals withheld tools mid-run.
  - load programmatically:
    - from verifiers.envs.bfcl_v3_env import load_environment
    - env = load_environment(version="v3", mode="multi", dataset_file="path/to.jsonl")

- v3 (single-turn): BFCLV3SingleTurnEnv (ToolEnv, max_turns=1)
  - “Next-action” probes derived from v3 trajectories; binary reward (execution equivalence).
  - Instantiate directly or via load_environment(..., mode="single").

- v4 (multi-turn): BFCLV4WebEnv (ToolEnv)
  - Tools: duckduckgo_search(keywords, max_results=10, region="wt-wt"); fetch_url_content(url, mode in {"raw","markdown","truncate"}).
  - Seeded failure injection on fetch; optional offline cache for determinism.
  - load programmatically:
    - from verifiers.envs.bfcl_v4_env import load_environment
    - env = load_environment(version="v4", mode="multi", offline_cache_dir="./cache", fetch_fail_rate=0.2, failure_seed=1234)

- v4 (single-turn):
  - BFCLV4SingleTurnEnv (B1): ToolEnv with max_turns=1.
  - BFCLV4OracleSingleTurnEnv (B2): SingleTurnEnv with oracle evidence; no tools.

- Rubrics:
  - v4 uses BFCLV4Rubric (normalized exact-match on final JSON {"answer","context"}; grades only "answer").
  - v3 rubrics attach externally; tests demonstrate binary execution success.

Data loading
- Read JSONL manually per BFCL instructions (do not call datasets.load_dataset).
- Normalize to Dataset with prompt (messages), answer (v4), and info metadata.

Parallel tool calls
- ToolEnv supports multiple tool calls in a single assistant message (v4 single-turn B1).

Reproducibility
- v4: set failure_seed and offline_cache_dir; offline=True when cache dir provided and no network.
- v3: reset internal state per rollout.
