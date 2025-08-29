
#!/usr/bin/env python3

"""
Test script to verify that the network safety opt-in flag works correctly.
"""

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv, BFCLV4SingleTurnEnv, V4Config
from datasets import Dataset

def test_network_safety_flag():
    """Test that the enforce_network_safety flag works correctly."""
    
    # Create a simple dataset
    ds = Dataset.from_dict({
        'prompt': [{'role': 'user', 'content': 'test'}],
        'answer': ['test'],
        'info': [{}]
    })
    
    # Test with enforce_network_safety=True (default)
    env_safe = BFCLV4WebEnv(dataset=ds, enforce_network_safety=True)
    assert env_safe.config.enforce_network_safety == True, "Default should be True"
    
    # Test with enforce_network_safety=False
    env_unsafe = BFCLV4WebEnv(dataset=ds, enforce_network_safety=False)
    assert env_unsafe.config.enforce_network_safety == False, "Should be set to False"
    
    # Test with BFCLV4SingleTurnEnv
    env_single_safe = BFCLV4SingleTurnEnv(dataset=ds, enforce_network_safety=True)
    assert env_single_safe.config.enforce_network_safety == True, "Single turn default should be True"
    
    env_single_unsafe = BFCLV4SingleTurnEnv(dataset=ds, enforce_network_safety=False)
    assert env_single_unsafe.config.enforce_network_safety == False, "Single turn should be set to False"
    
    print("✓ Network safety flag tests passed")

def test_url_safety_function():
    """Test that the _is_allowed_url function respects the enforce_network_safety flag."""
    
    from verifiers.envs.bfcl_v4_env import _is_allowed_url
    
    # Test URLs that should be blocked when enforce_network_safety=True
    blocked_urls = [
        "http://127.0.0.1:8080",  # loopback
        "http://192.168.1.1",     # private IP
        "http://localhost:8000",  # localhost
        "ftp://example.com",      # non-HTTP(S)
        "gopher://test.com",      # non-HTTP(S)
    ]
    
    # Test URLs that should be allowed even when enforce_network_safety=True
    allowed_urls = [
        "http://example.com",
        "https://google.com",
        "http://github.com/openai",
    ]
    
    # Test with enforce_network_safety=True (default)
    for url in blocked_urls:
        result = _is_allowed_url(url, enforce_network_safety=True)
        assert result == False, f"URL {url} should be blocked when enforce_network_safety=True"
    
    for url in allowed_urls:
        result = _is_allowed_url(url, enforce_network_safety=True)
        assert result == True, f"URL {url} should be allowed when enforce_network_safety=True"
    
    # Test with enforce_network_safety=False
    for url in blocked_urls + allowed_urls:
        result = _is_allowed_url(url, enforce_network_safety=False)
        assert result == True, f"URL {url} should be allowed when enforce_network_safety=False"
    
    print("✓ URL safety function tests passed")

if __name__ == "__main__":
    test_network_safety_flag()
    test_url_safety_function()
    print("All network safety tests passed!")
