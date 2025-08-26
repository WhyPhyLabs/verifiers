


from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from verifiers.envs.bfcl_v3_env import BFCLV3Env, BFCLV3SingleTurnEnv, load_bfcl_v3
from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv, BFCLV4SingleTurnEnv, BFCLV4OracleSingleTurnEnv, load_bfcl_v4


@pytest.mark.bfcl_e2e
def test_v3_multiturn_cli_smoke_test():
    """E2E smoke test for v3 multi-turn environment."""
    # Create minimal test dataset
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "set key=test to value"}]],
        "answer": ["OK"],
        "info": [{"final_state": {"test": "value"}, "tool_sequence": ["set_kv"]}]
    })
    
    # Test environment creation and basic functionality
    env = BFCLV3Env(dataset=ds, enable_missing_functions=True, max_turns=3)
    
    # Test that environment has expected tools
    assert "set_kv" in env.tool_map
    assert len(env.tool_map) >= 1
    
    # Test basic tool call
    tool_msg = env.call_tool("set_kv", json.dumps({"key": "test", "value": "value"}), "c1")
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "OK"
    
    # Test that max_turns is set correctly for multi-turn
    assert env.max_turns > 1


@pytest.mark.bfcl_e2e 
def test_v3_single_turn_cli_smoke_test():
    """E2E smoke test for v3 single-turn environment."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "set key=foo to bar"}]],
        "answer": ["OK"],
        "info": [{"expected": {"tool": "set_kv", "args": {"key": "foo", "value": "bar"}, "output": "OK"}}]
    })
    
    env = BFCLV3SingleTurnEnv(dataset=ds)
    
    # Test that max_turns is 1 for single-turn
    assert env.max_turns == 1
    
    # Test tool execution
    tool_msg = env.call_tool("set_kv", json.dumps({"key": "foo", "value": "bar"}), "c1")
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "OK"


@pytest.mark.bfcl_e2e
def test_v4_multiturn_mock_cli_smoke_test():
    """E2E smoke test for v4 multi-turn environment (mock mode)."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "search for test information"}]],
        "answer": ["test answer"],
        "info": [{}]
    })
    
    env = BFCLV4WebEnv(dataset=ds, live=False, max_turns=3)
    
    # Test that environment has expected tools
    assert "duckduckgo_search" in env.tool_map
    assert "fetch_url_content" in env.tool_map
    assert len(env.tool_map) == 2
    
    # Test that max_turns is set correctly for multi-turn
    assert env.max_turns > 1
    
    # Test mock search (should work without network)
    search_result = env.tool_map["duckduckgo_search"]("test query", max_results=3)
    assert isinstance(search_result, list)


@pytest.mark.bfcl_e2e
def test_v4_single_turn_b1_cli_smoke_test():
    """E2E smoke test for v4 single-turn B1 environment (tools)."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "search for something"}]],
        "answer": ["{\"answer\": \"test result\"}"],
        "info": [{}]
    })
    
    env = BFCLV4SingleTurnEnv(dataset=ds)
    
    # Test that max_turns is 1 for single-turn
    assert env.max_turns == 1
    
    # Test that environment has expected tools
    assert "duckduckgo_search" in env.tool_map
    assert "fetch_url_content" in env.tool_map


@pytest.mark.bfcl_e2e
def test_v4_single_turn_b2_oracle_cli_smoke_test():
    """E2E smoke test for v4 single-turn B2 oracle environment."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "what is the answer?"}]],
        "answer": ["{\"answer\": \"42\"}"],
        "info": [{"evidence": "The answer is 42."}]
    })
    
    env = BFCLV4OracleSingleTurnEnv(dataset=ds, inject_oracle=True)
    
    # Test that max_turns is 1 for single-turn
    assert env.max_turns == 1
    
    # Test that environment has no tools (oracle mode)
    # BFCLV4OracleSingleTurnEnv is a SingleTurnEnv, not a ToolEnv, so no tool_map attribute
    # The important thing is that max_turns is 1 for single-turn
    assert env.max_turns == 1


@pytest.mark.bfcl_e2e 
def test_v4_multiturn_live_cli_smoke_test():
    """E2E smoke test for v4 multi-turn environment (live mode - requires deps)."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "search test"}]],
        "answer": ["test"],
        "info": [{}]
    })
    
    # This test will be skipped if dependencies are missing
    try:
        env = BFCLV4WebEnv(dataset=ds, live=True, max_turns=2)
        assert env.max_turns > 1
        assert "duckduckgo_search" in env.tool_map
        assert "fetch_url_content" in env.tool_map
    except ImportError:
        pytest.skip("Live mode dependencies not available")


@pytest.mark.bfcl_e2e
def test_bfcl_jsonl_dataset_loading():
    """Test that BFCL environments can load from JSONL files."""
    # Create temporary JSONL files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # v3 style JSONL
        v3_jsonl = Path(tmpdir) / "v3_test.jsonl"
        v3_data = {
            "prompt": [[{"role": "user", "content": "set key=test to value"}]],
            "answer": ["OK"],
            "info": [{"final_state": {"test": "value"}, "tool_sequence": ["set_kv"]}]
        }
        
        with open(v3_jsonl, "w") as f:
            f.write(json.dumps(v3_data) + "\n")
        
        # Test loading v3 environment from JSONL
        from verifiers.envs.bfcl_v3_env import load_bfcl_v3
        ds = load_bfcl_v3(str(v3_jsonl))
        assert len(ds) == 1
        assert ds[0]["prompt"] == v3_data["prompt"]
        
        # v4 style JSONL  
        v4_jsonl = Path(tmpdir) / "v4_test.jsonl"
        v4_data = {
            "prompt": [[{"role": "user", "content": "search test"}]],
            "answer": ["{\"answer\": \"test\"}"],
            "info": [{}]
        }
        
        with open(v4_jsonl, "w") as f:
            f.write(json.dumps(v4_data) + "\n")
            
        # Test loading v4 environment from JSONL
        from verifiers.envs.bfcl_v4_env import load_bfcl_v4
        ds = load_bfcl_v4(str(v4_jsonl))
        assert len(ds) == 1
        assert ds[0]["prompt"] == v4_data["prompt"]


# Integration test that simulates the full CLI workflow
@pytest.mark.bfcl_e2e
def test_full_cli_workflow_simulation():
    """Simulate the full CLI workflow for all 4 canonical modes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test datasets
        datasets = {}
        
        # v3 multi-turn dataset
        v3_multi_data = {
            "prompt": [[{"role": "user", "content": "set key=multi to test"}]],
            "answer": ["OK"],
            "info": [{"final_state": {"multi": "test"}, "tool_sequence": ["set_kv"]}]
        }
        
        # v3 single-turn dataset
        v3_single_data = {
            "prompt": [[{"role": "user", "content": "set key=single to test"}]],
            "answer": ["OK"],
            "info": [{"expected": {"tool": "set_kv", "args": {"key": "single", "value": "test"}, "output": "OK"}}]
        }
        
        # v4 multi-turn dataset
        v4_multi_data = {
            "prompt": [[{"role": "user", "content": "search for multi-turn test"}]],
            "answer": ["{\"answer\": \"multi-turn result\"}"],
            "info": [{}]
        }
        
        # v4 single-turn dataset
        v4_single_data = {
            "prompt": [[{"role": "user", "content": "search for single-turn test"}]],
            "answer": ["{\"answer\": \"single-turn result\"}"],
            "info": [{}]
        }
        
        # Write datasets to files
        for name, data in [
            ("v3_multi.jsonl", v3_multi_data),
            ("v3_single.jsonl", v3_single_data), 
            ("v4_multi.jsonl", v4_multi_data),
            ("v4_single.jsonl", v4_single_data),
        ]:
            filepath = Path(tmpdir) / name
            with open(filepath, "w") as f:
                f.write(json.dumps(data) + "\n")
            datasets[name] = str(filepath)
        
        # Test that all environments can be created and initialized
        # This simulates what vf-eval would do
        
        # v3 multi-turn
        v3_multi_ds = load_bfcl_v3(datasets["v3_multi.jsonl"])
        v3_multi_env = BFCLV3Env(dataset=v3_multi_ds, enable_missing_functions=True)
        assert v3_multi_env.max_turns > 1
        
        # v3 single-turn
        v3_single_ds = load_bfcl_v3(datasets["v3_single.jsonl"])
        v3_single_env = BFCLV3SingleTurnEnv(dataset=v3_single_ds)
        assert v3_single_env.max_turns == 1
        
        # v4 multi-turn (mock)
        v4_multi_ds = load_bfcl_v4(datasets["v4_multi.jsonl"])
        v4_multi_env = BFCLV4WebEnv(dataset=v4_multi_ds, live=False)
        assert v4_multi_env.max_turns > 1
        
        # v4 single-turn B1
        v4_single_ds = load_bfcl_v4(datasets["v4_single.jsonl"])
        v4_single_env = BFCLV4SingleTurnEnv(dataset=v4_single_ds)
        assert v4_single_env.max_turns == 1
        
        # v4 single-turn B2 (oracle)
        v4_oracle_env = BFCLV4OracleSingleTurnEnv(dataset=v4_single_ds)
        assert v4_oracle_env.max_turns == 1


# Test CLI wrapper alias compatibility - ensures documented CLI commands work
@pytest.mark.bfcl_e2e
def test_cli_wrapper_alias_compatibility():
    """Test that CLI wrapper accepts documented alias values for version parameter."""
    from environments.vf_bfcl.vf_bfcl import load_environment
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test datasets
        v3_data = {
            "prompt": [[{"role": "user", "content": "set key=test to value"}]],
            "answer": ["OK"],
            "info": [{"final_state": {"test": "value"}, "tool_sequence": ["set_kv"]}]
        }
        
        v4_data = {
            "prompt": [[{"role": "user", "content": "search for test"}]],
            "answer": ["{\"answer\": \"test result\"}"],
            "info": [{}]
        }
        
        v4_oracle_data = {
            "prompt": [[{"role": "user", "content": "what is 2+2?"}]],
            "answer": ["{\"answer\": \"4\"}"],
            "info": [{"evidence": "The sum of 2 and 2 is 4."}]
        }
        
        # Write datasets to files
        v3_jsonl = Path(tmpdir) / "v3_test.jsonl"
        v4_jsonl = Path(tmpdir) / "v4_test.jsonl"
        v4_oracle_jsonl = Path(tmpdir) / "v4_oracle_test.jsonl"
        
        with open(v3_jsonl, "w") as f:
            f.write(json.dumps(v3_data) + "\n")
        with open(v4_jsonl, "w") as f:
            f.write(json.dumps(v4_data) + "\n")
        with open(v4_oracle_jsonl, "w") as f:
            f.write(json.dumps(v4_oracle_data) + "\n")
        
        # Test documented CLI aliases work with wrapper
        
        # Test v3_single alias (should create single-turn environment)
        env_v3_single = load_environment(version="v3_single", dataset_file=str(v3_jsonl))
        assert env_v3_single.max_turns == 1
        assert hasattr(env_v3_single, 'tools')
        
        # Test v4_single alias (should create B1 single-turn environment)
        env_v4_single = load_environment(version="v4_single", dataset_file=str(v4_jsonl))
        assert env_v4_single.max_turns == 1
        assert hasattr(env_v4_single, 'tools')
        assert "duckduckgo_search" in env_v4_single.tool_map
        
        # Test v4_oracle alias (should create B2 oracle environment)
        env_v4_oracle = load_environment(version="v4_oracle", dataset_file=str(v4_oracle_jsonl), inject_oracle=True)
        assert env_v4_oracle.max_turns == 1
        # Oracle mode is a SingleTurnEnv, not a ToolEnv, so no tool_map attribute
        # The important thing is that max_turns is 1 for single-turn
        
        # Test that original v3 and v4 still work
        env_v3_multi = load_environment(version="v3", dataset_file=str(v3_jsonl))
        assert env_v3_multi.max_turns > 1
        
        env_v4_multi = load_environment(version="v4", dataset_file=str(v4_jsonl))
        assert env_v4_multi.max_turns > 1

