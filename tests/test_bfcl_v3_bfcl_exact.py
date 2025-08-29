
from __future__ import annotations

import json
from datasets import Dataset

from verifiers.envs.bfcl_v3_env import BFCLV3Env


def test_v3_ordered_subsequence_positive():
    """Test that ordered subsequence matching works correctly (positive case)."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Complete task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": ["set_kv", "secret_add"]}]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Test case: actual sequence contains expected sequence as ordered subsequence
    # Actual: ["get_kv", "set_kv", "get_kv", "secret_add", "set_kv"]
    # Expected: ["set_kv", "secret_add"] -> should pass
    actual_sequence = ["get_kv", "set_kv", "get_kv", "secret_add", "set_kv"]
    expected_sequence = ["set_kv", "secret_add"]
    
    result = env._is_ordered_subsequence(expected_sequence, actual_sequence)
    assert result is True, f"Expected {expected_sequence} to be ordered subsequence of {actual_sequence}"


def test_v3_ordered_subsequence_negative():
    """Test that ordered subsequence matching fails correctly (negative case)."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Complete task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": ["set_kv", "secret_add"]}]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Test case: actual sequence does NOT contain expected sequence as ordered subsequence
    # Actual: ["set_kv", "get_kv", "set_kv"] 
    # Expected: ["set_kv", "secret_add"] -> should fail (secret_add missing)
    actual_sequence = ["set_kv", "get_kv", "set_kv"]
    expected_sequence = ["set_kv", "secret_add"]
    
    result = env._is_ordered_subsequence(expected_sequence, actual_sequence)
    assert result is False, f"Expected {expected_sequence} NOT to be ordered subsequence of {actual_sequence}"


def test_v3_ordered_subsequence_empty():
    """Test that empty sequences are handled correctly."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Complete task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": []}]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Empty expected sequence should always pass
    actual_sequence = ["set_kv", "secret_add"]
    expected_sequence = []
    
    result = env._is_ordered_subsequence(expected_sequence, actual_sequence)
    assert result is True, "Empty expected sequence should always pass"


def test_v3_termination_last_turn_success():
    """Test that success on the last allowed turn (turn == max_turns) passes."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Complete task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}]
    })
    env = BFCLV3Env(dataset=ds, max_turns=3)
    
    # Simulate state where turn == max_turns (should NOT fail)
    state = {
        "turn": 3,  # Exactly equal to max_turns
        "kv": {"result": "success"},
        "tool_names": ["set_kv"]
    }
    info = {"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}
    
    success = env._determine_v3_success([], [], state, info)
    assert success is True, "Success on last allowed turn should pass"


def test_v3_termination_exceeds_max_turns_failure():
    """Test that exceeding max turns (turn > max_turns) fails."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Complete task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}]
    })
    env = BFCLV3Env(dataset=ds, max_turns=3)
    
    # Simulate state where turn > max_turns (should fail)
    state = {
        "turn": 4,  # Exceeds max_turns
        "kv": {"result": "success"},
        "tool_names": ["set_kv"]
    }
    info = {"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}
    
    success = env._determine_v3_success([], [], state, info)
    assert success is False, "Exceeding max turns should fail"


def test_v3_per_turn_success_all_pass():
    """Test per-turn success when all turns pass both checks."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Multi-turn task"}]], 
        "info": [{
            "turns": [
                {
                    "final_state": {"foo": "bar"},
                    "required_tools": ["set_kv"]
                },
                {
                    "final_state": {"result": 5},
                    "required_tools": ["secret_add"]
                }
            ]
        }]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Simulate successful per-turn execution
    state = {
        "turn": 2,
        "turn_states": [
            {"start": {}, "end": {"foo": "bar"}},
            {"start": {"foo": "bar"}, "end": {"foo": "bar", "result": 5}}
        ],
        "turn_tool_sequences": [
            ["set_kv"],
            ["secret_add"]
        ]
    }
    info = {
        "turns": [
            {"final_state": {"foo": "bar"}, "required_tools": ["set_kv"]},
            {"final_state": {"result": 5}, "required_tools": ["secret_add"]}
        ]
    }
    
    success = env._check_per_turn_success(state, info)
    assert success is True, "All turns passing both checks should succeed"


def test_v3_per_turn_failure_one_turn_fails():
    """Test per-turn failure when one turn fails a check."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Multi-turn task"}]], 
        "info": [{
            "turns": [
                {
                    "final_state": {"foo": "bar"},
                    "required_tools": ["set_kv"]
                },
                {
                    "final_state": {"result": 5},
                    "required_tools": ["secret_add"]
                }
            ]
        }]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Simulate second turn failing state check
    state = {
        "turn": 2,
        "turn_states": [
            {"start": {}, "end": {"foo": "bar"}},
            {"start": {"foo": "bar"}, "end": {"foo": "bar", "result": "wrong"}}  # Wrong result
        ],
        "turn_tool_sequences": [
            ["set_kv"],
            ["secret_add"]
        ]
    }
    info = {
        "turns": [
            {"final_state": {"foo": "bar"}, "required_tools": ["set_kv"]},
            {"final_state": {"result": 5}, "required_tools": ["secret_add"]}
        ]
    }
    
    success = env._check_per_turn_success(state, info)
    assert success is False, "One turn failing should cause overall failure"


def test_v3_per_turn_fallback_to_global():
    """Test that per-turn falls back to global checking when per-turn data missing."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Simple task"}]], 
        "info": [{"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Simulate state without per-turn data
    state = {
        "turn": 1,
        "kv": {"result": "success"},
        "tool_names": ["set_kv"]
    }
    info = {"final_state": {"result": "success"}, "tool_sequence": ["set_kv"]}
    
    success = env._determine_v3_success([], [], state, info)
    assert success is True, "Should fall back to global checking when per-turn data missing"


def test_v3_turn_state_goal_private_keys_ignored():
    """Test that private keys (starting with _) are ignored in turn state checking."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Task with private state"}]], 
        "info": [{
            "turns": [
                {
                    "final_state": {"foo": "bar"},
                    "required_tools": ["set_kv"]
                }
            ]
        }]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Simulate turn state with private keys that don't match
    turn_state = {
        "start": {},
        "end": {"foo": "bar", "_private": "secret"}  # Private key present but not in expected
    }
    turn_spec = {"final_state": {"foo": "bar"}, "required_tools": ["set_kv"]}
    
    success = env._check_turn_state_goal(turn_state, turn_spec)
    assert success is True, "Private keys should be ignored in turn state checking"


def test_v3_turn_tool_sequence_subsequence():
    """Test that turn tool sequence checking uses subsequence matching."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "Task with extra tools"}]], 
        "info": [{
            "turns": [
                {
                    "final_state": {"foo": "bar"},
                    "required_tools": ["set_kv"]
                }
            ]
        }]
    })
    env = BFCLV3Env(dataset=ds)
    
    # Simulate turn with extra tools before required tool
    turn_tools = ["get_kv", "set_kv", "get_kv"]  # Extra get_kv calls
    turn_spec = {"final_state": {"foo": "bar"}, "required_tools": ["set_kv"]}
    
    success = env._check_turn_tool_sequence(turn_tools, turn_spec)
    assert success is True, "Turn tool sequence should allow extra tools (subsequence matching)"
