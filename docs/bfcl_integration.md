# BFCL Integration (v3 and v4)

## v3
- Multi-turn: BFCLV3Env (ToolEnv)
  - Missing-Functions gating (attempted unknown tool + broader textual triggers)
  - State recorded in `state.kv` and tool sequence in `state.tool_names`
  - Rubric (multi-turn): BFCLV3MultiTurnRubric
    - `_multiturn_state_goal`: checks `info.final_state` against `state.kv`
    - `_multiturn_sequence_match`: checks `info.tool_sequence` prefix vs `state.tool_names`
- Single-turn: BFCLV3SingleTurnEnv (ToolEnv, max_turns=1)
  - Rubric (single-turn): BFCLV3SingleTurnRubric (exec-equivalence via `info.expected.output`)

## v4
- Tools: `duckduckgo_search(keywords, max_results=10, region)` and `fetch_url_content(url, mode)`
- Modes: mock (default) and live (`live=True`)
  - Mock: deterministic search + fetch, seeded failure injection, offline cache
  - Live: optional extras (`verifiers[bfcl]`), httpx timeouts/retries, URL allowlist, simple HTML→text via BeautifulSoup
- Modular Search API
  - Interface: `SearchClient` with `.search(keywords, max_results, region)`
  - Implementations: `MockSearchClient` and `DDGSearchClient`; you can inject your own (e.g., Bing, SerpAPI)
- Single-turn variants
  - B1: BFCLV4SingleTurnEnv (ToolEnv, max_turns=1)
  - B2: BFCLV4OracleSingleTurnEnv (SingleTurnEnv, max_turns=1) with optional oracle evidence injection (`info.evidence`)
- Rubric: BFCLV4Rubric (normalized exact-match over strict JSON `{"answer": ...}`)

## Security & Determinism
- URL allowlist: only http/https; block private/reserved IPs
- Retries/backoff and timeouts for live fetch
- Offline cache and seeded failure injection for reproducibility

## Usage
```python
from verifiers.envs import bfcl_v3_env as v3, bfcl_v4_env as v4

# v3 multi-turn
env = v3.load_environment(version="v3", mode="multi", dataset_file="v3.jsonl")
# v4 multi-turn live with modular search
env = v4.load_environment(version="v4", mode="multi", dataset_file="v4.jsonl", live=True)
```
