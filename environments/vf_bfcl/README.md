

# VF-BFCL Environment

This is the BFCL (Berkeley Function Calling Leaderboard) environment for the verifiers framework.

## Installation

### From the verifiers repository

If you have the verifiers repository checked out, you can install this environment with:

```bash
cd environments/vf_bfcl
pip install -e .
```

### With bfcl dependencies

To use live mode features (web search, URL fetching), install with the bfcl extras:

```bash
pip install 'verifiers[bfcl]'
```

## Usage

The BFCL environment supports both v3 and v4 variants, with single-turn and multi-turn modes:

### BFCL v3 (Multi-turn, Stateful)

```python
from vf_bfcl import load_bfcl_env

# Multi-turn v3 environment
env = load_bfcl_env("v3", dataset=dataset, enable_missing_functions=True)

# Single-turn v3 environment  
env = load_bfcl_env("v3_single", dataset=dataset)
```

### BFCL v4 (Web Search, Agentic)

```python
from vf_bfcl import load_bfcl_env

# Multi-turn v4 environment (mock mode)
env = load_bfcl_env("v4", dataset=dataset, live=False)

# Multi-turn v4 environment (live mode - requires bfcl extras)
env = load_bfcl_env("v4", dataset=dataset, live=True)

# Single-turn v4 B1 (tools)
env = load_bfcl_env("v4_single", dataset=dataset)

# Single-turn v4 B2 (oracle)
env = load_bfcl_env("v4_oracle", dataset=dataset)
```

## Features

### BFCL v3
- Multi-turn stateful evaluation
- Missing-function gating with textual and attempted-call triggers
- State-goal and sequence-prefix rubrics with configurable weights

### BFCL v4
- Web search and URL fetching tools
- URL allowlist for security (blocks private/reserved IPs)
- Mock and live modes
- Oracle variant for single-turn evaluation
- Unicode-aware answer normalization

## CLI Usage

Use with vf-eval:

```bash
# v3 multi-turn
vf-eval --env vf_bfcl --env-config '{"version": "v3", "enable_missing_functions": true}' --dataset bfcl_v3.jsonl

# v3 single-turn  
vf-eval --env vf_bfcl --env-config '{"version": "v3_single"}' --dataset bfcl_v3.jsonl

# v4 multi-turn (mock)
vf-eval --env vf_bfcl --env-config '{"version": "v4", "live": false}' --dataset bfcl_v4.jsonl

# v4 multi-turn (live)
vf-eval --env vf_bfcl --env-config '{"version": "v4", "live": true}' --dataset bfcl_v4.jsonl

# v4 single-turn B1
vf-eval --env vf_bfcl --env-config '{"version": "v4_single"}' --dataset bfcl_v4_b1.jsonl

# v4 single-turn B2 (oracle)
vf-eval --env vf_bfcl --env-config '{"version": "v4_oracle"}' --dataset bfcl_v4_b2.jsonl
```

## Testing

Run tests with:

```bash
pytest tests/
```

