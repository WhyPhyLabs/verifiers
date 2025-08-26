from __future__ import annotations

import json

from datasets import Dataset

from verifiers.envs.bfcl_v3_env import BFCLV3Env, BFCLV3SingleTurnEnv


def test_v3_missing_functions_gating_reveals_tool_and_injects_doc():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "start"}]], "info": [{}]})
    env = BFCLV3Env(dataset=ds, enable_missing_functions=True)
    # Assistant indicates the need for missing function (no tool_calls)
    assistant_msg = [{"role": "assistant", "content": "I need a missing function to proceed."}]
    env_msgs, _ = env.env_response(assistant_msg, {"turn": 1})
    assert any(m["role"] == "user" and "New tool unlocked" in m["content"] for m in env_msgs)
    assert "secret_add" in env.tool_map
    # Issue the tool call now that it's revealed
    tool_msg = env.call_tool("secret_add", json.dumps({"x": 2, "y": 3}), "c1")
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "5"


def test_v3_missing_functions_gating_benign_text_no_reveal():
    """Test that benign assistant text does NOT reveal the missing tool."""
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "start"}]], "info": [{}]})
    env = BFCLV3Env(dataset=ds, enable_missing_functions=True)
    
    # Assistant says benign things that don't trigger reveal
    benign_messages = [
        [{"role": "assistant", "content": "I need to calculate something."}],
        [{"role": "assistant", "content": "Let me think about this problem."}],
        [{"role": "assistant", "content": "I can help you with that."}],
        [{"role": "assistant", "content": "The answer is 42."}],
        [{"role": "assistant", "content": "I need to use the available tools."}],
    ]
    
    for assistant_msg in benign_messages:
        # Reset the environment for each test
        env = BFCLV3Env(dataset=ds, enable_missing_functions=True)
        
        # This should NOT reveal the tool
        env_msgs, state = env.env_response(assistant_msg, {"turn": 1})
        
        # No user messages should be added
        user_msgs = [m for m in env_msgs if m["role"] == "user"]
        assert len(user_msgs) == 0, f"Benign message should not reveal tool: {assistant_msg[0]['content']}"
        
        # Tool should still be hidden
        assert "secret_add" not in env.tool_map
        assert not env._revealed


def test_v3_single_turn_next_action_env_builds_and_executes():
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "set key=foo to bar"}]],
            "info": [{"expected": {"tool": "set_kv", "args": {"key": "foo", "value": "bar"}, "output": "OK"}}],
        }
    )
    env = BFCLV3SingleTurnEnv(dataset=ds)
    assert env.max_turns == 1
    tool_msg = env.call_tool("set_kv", json.dumps({"key": "foo", "value": "bar"}), "c1")
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "OK"
