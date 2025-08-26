# BFCL Integration (v3 and v4)

## v3
- Multi-turn: BFCLV3Env (ToolEnv)
  - Missing-Functions gating (attempted unknown tool + broader textual triggers)
  - State recorded in `state.kv` and tool sequence in `state.tool_names`
  - Rubric (multi-turn): BFCLV3MultiTurnRubric
    - `_multiturn_state_goal`: checks `info.final_state` against `state.kv` (weight: 0.7)
    - `_multiturn_sequence_match`: checks `info.tool_sequence` prefix vs `state.tool_names` (weight: 0.3)
    - **Note**: Multi-turn scoring combines state goal achievement (70%) and tool sequence prefix matching (30%)
- Single-turn: BFCLV3SingleTurnEnv (ToolEnv, max_turns=1)
  - Rubric (single-turn): BFCLV3SingleTurnRubric (exec-equivalence via `info.expected.output`)
  - **Note**: Single-turn scoring follows Tool-N1's binary functional-correctness approach, comparing tool output against expected results

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
  - **Note**: v4 grading uses only the `answer` field from the JSON response, matching the BFCL v4 specification's metric. Context and other fields are ignored.
  - **Normalization**: Includes Unicode NFKC normalization, whitespace trimming, diacritic removal, and dash normalization for robust matching.
  - **Score Modes**: 
    - `"exec"` (default): Execution-equivalence scoring for functional correctness
    - `"ast"`: AST-based scoring for leaderboard parity with some BFCL evaluations

## Security & Determinism
- URL allowlist: only http/https; block private/reserved IPs
- Retries/backoff and timeouts for live fetch
- Offline cache and seeded failure injection for reproducibility

## Usage

### Python API

```python
from verifiers.envs import bfcl_v3_env as v3, bfcl_v4_env as v4

# v3 multi-turn
env = v3.load_environment(version="v3", mode="multi", dataset_file="v3.jsonl")
# v4 multi-turn live with modular search
env = v4.load_environment(version="v4", mode="multi", dataset_file="v4.jsonl", live=True)
```

### CLI Commands (Copy-Paste Ready)

#### Installation

First, install the package with BFCL dependencies:

```bash
# Install with BFCL extras for live mode
pip install 'verifiers[bfcl]'

# Or install from repository
git clone https://github.com/WhyPhyLabs/verifiers
cd verifiers
pip install -e '.[bfcl]'

# Install the BFCL environment module
cd environments/vf_bfcl
pip install -e .
```

#### v3 Multi-Turn Mode

```bash
# Run v3 multi-turn evaluation with missing-functions gating
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v3", "enable_missing_functions": true}' \
  --dataset path/to/bfcl_v3_dataset.jsonl \
  --output results/v3_multi_turn.jsonl

# With custom max_turns
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v3", "enable_missing_functions": true, "max_turns": 5}' \
  --dataset path/to/bfcl_v3_dataset.jsonl \
  --output results/v3_multi_turn.jsonl
```

#### v3 Single-Turn Mode

```bash
# Run v3 single-turn evaluation (binary execution equivalence)
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v3_single"}' \
  --dataset path/to/bfcl_v3_dataset.jsonl \
  --output results/v3_single_turn.jsonl
```

#### v4 Multi-Turn Mode (Mock)

```bash
# Run v4 multi-turn evaluation with mock search/fetch
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": false}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_multi_turn_mock.jsonl

# With custom timeout and retry settings
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": false, "timeout_s": 10.0, "max_retries": 3}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_multi_turn_mock.jsonl
```

#### v4 Multi-Turn Mode (Live)

```bash
# Run v4 multi-turn evaluation with live search/fetch (requires bfcl extras)
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": true}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_multi_turn_live.jsonl

# With custom search client and failure injection
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": true, "fetch_fail_rate": 0.1, "failure_seed": 42}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_multi_turn_live.jsonl
```

#### v4 Single-Turn Mode (B1 - Tools)

```bash
# Run v4 single-turn evaluation with tools (B1 split)
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4_single"}' \
  --dataset path/to/bfcl_v4_b1_dataset.jsonl \
  --output results/v4_single_turn_b1.jsonl
```

#### v4 Single-Turn Mode (B2 - Oracle)

```bash
# Run v4 single-turn evaluation with oracle evidence (B2 split)
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4_oracle", "inject_oracle": true}' \
  --dataset path/to/bfcl_v4_b2_dataset.jsonl \
  --output results/v4_single_turn_b2.jsonl

# Without oracle injection
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4_oracle", "inject_oracle": false}' \
  --dataset path/to/bfcl_v4_b2_dataset.jsonl \
  --output results/v4_single_turn_b2_no_oracle.jsonl
```

### Advanced CLI Usage

#### Parallel Evaluation

```bash
# Run multiple configurations in parallel
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v3", "enable_missing_functions": true}' \
  --dataset path/to/bfcl_v3_dataset.jsonl \
  --output results/v3_multi_turn.jsonl \
  --num-workers 4 \
  --batch-size 32
```

#### Custom Search Client

```bash
# Use custom search client (implement SearchClient interface)
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": false, "search_client": "custom_search_client"}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_custom_search.jsonl
```

#### Offline Cache

```bash
# Use offline cache for reproducible results
vf-eval \
  --env vf_bfcl \
  --env-config '{"version": "v4", "live": false, "offline_cache_dir": "/path/to/cache"}' \
  --dataset path/to/bfcl_v4_dataset.jsonl \
  --output results/v4_cached.jsonl
```

### Environment Configuration Options

#### v3 Configuration
```json
{
  "version": "v3" | "v3_single",
  "enable_missing_functions": true | false,
  "max_turns": 8,
  "withheld_tool_name": "secret_add"
}
```

#### v4 Configuration
```json
{
  "version": "v4" | "v4_single" | "v4_oracle",
  "live": true | false,
  "max_turns": 8,
  "timeout_s": 8.0,
  "max_retries": 2,
  "fetch_fail_rate": 0.0,
  "failure_seed": 1234,
  "offline_cache_dir": "/path/to/cache",
  "include_snippets": true,
  "inject_oracle": true | false
}
```

### Dataset Format Examples

#### v3 Dataset Format
```json
{"prompt": [[{"role": "user", "content": "Set key=foo to bar"}]], "answer": "OK", "info": {"final_state": {"foo": "bar"}, "tool_sequence": ["set_kv"]}}
{"prompt": [[{"role": "user", "content": "Add 2 and 3"}]], "answer": "5", "info": {"final_state": {"result": 5}, "tool_sequence": ["secret_add"]}}
```

#### v4 Dataset Format
```json
{"prompt": [[{"role": "user", "content": "What is the capital of France?"}]], "answer": "{\"answer\": \"Paris\"}", "info": {}}
{"prompt": [[{"role": "user", "content": "Search for AI news"}]], "answer": "{\"answer\": \"Latest AI developments...\"}", "info": {"sources": ["example.com"]}}
```

#### v4 Oracle Dataset Format
```json
{"prompt": [[{"role": "user", "content": "What is 2+2?"}]], "answer": "{\"answer\": \"4\"}", "info": {"evidence": "The sum of 2 and 2 is 4."}}
```
