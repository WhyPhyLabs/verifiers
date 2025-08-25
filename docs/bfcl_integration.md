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
  - Tools (deterministic stubs for offline reproducibility):
    - duckduckgo_search(keywords, max_results=10, region="wt-wt") -> list[{title,url,snippet?}]
    - fetch_url_content(url, mode in {"raw","markdown","truncate"}) -> str
  - Failure injection (seeded) returns canonical error strings (never raises) to keep loops alive.
  - Offline cache semantics: if offline_cache_dir is set and key missing, search returns [] and fetch returns "".
  - load programmatically:
    - from verifiers.envs.bfcl_v4_env import load_environment
    - env = load_environment(version="v4", mode="multi", offline_cache_dir="./cache", fetch_fail_rate=0.2, failure_seed=1234)

- v4 (single-turn):
  - BFCLV4SingleTurnEnv (B1): ToolEnv with max_turns=1 (parallel tool calls supported by ToolEnv).
  - BFCLV4OracleSingleTurnEnv (B2): SingleTurnEnv with max_turns=1 (oracle evidence; no tools).

- Rubrics:
  - v4 uses BFCLV4Rubric (normalized exact-match on final JSON {"answer","context"}; grades only "answer").
  - v3 rubrics attach externally; tests demonstrate binary execution success.

Data loading
- Read JSONL manually per BFCL instructions (do not call datasets.load_dataset).
- Normalize to Dataset with prompt (messages), answer (v4), and info metadata.

Parallel tool calls
- ToolEnv supports multiple tool calls in a single assistant message (v4 single-turn B1).

Reproducibility
- v4: set failure_seed and offline_cache_dir for deterministic behavior.
- v3: state reset per rollout.

CLI wrapper
- An installable wrapper is provided at `environments/vf_bfcl/vf_bfcl.py` so you can run via vf-eval after installing the environment module.
