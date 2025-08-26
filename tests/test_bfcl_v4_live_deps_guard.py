

from __future__ import annotations

import pytest
from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


def test_v4_live_mode_missing_deps_raises_importerror():
    """Test that live mode raises ImportError when dependencies are missing."""
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    
    # Mock the missing dependencies by temporarily setting them to None
    import verifiers.envs.bfcl_v4_env as bfcl_v4_module
    original_httpx = bfcl_v4_module.httpx
    original_ddgs = bfcl_v4_module.DDGS
    original_bs4 = bfcl_v4_module.BeautifulSoup
    
    try:
        # Simulate missing dependencies
        bfcl_v4_module.httpx = None
        bfcl_v4_module.DDGS = None
        bfcl_v4_module.BeautifulSoup = None
        
        # This should raise ImportError
        with pytest.raises(ImportError) as exc_info:
            BFCLV4WebEnv(dataset=ds, live=True)
        
        error_msg = str(exc_info.value)
        assert "Live mode requires additional dependencies" in error_msg
        assert "pip install 'verifiers[bfcl]'" in error_msg
        assert "httpx" in error_msg
        assert "duckduckgo-search" in error_msg
        assert "beautifulsoup4" in error_msg
        
    finally:
        # Restore original values
        bfcl_v4_module.httpx = original_httpx
        bfcl_v4_module.DDGS = original_ddgs
        bfcl_v4_module.BeautifulSoup = original_bs4


def test_v4_live_mode_partial_missing_deps_raises_importerror():
    """Test that live mode raises ImportError when some dependencies are missing."""
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    
    import verifiers.envs.bfcl_v4_env as bfcl_v4_module
    original_httpx = bfcl_v4_module.httpx
    
    try:
        # Simulate only httpx missing
        bfcl_v4_module.httpx = None
        
        with pytest.raises(ImportError) as exc_info:
            BFCLV4WebEnv(dataset=ds, live=True)
        
        error_msg = str(exc_info.value)
        assert "Live mode requires additional dependencies" in error_msg
        assert "httpx" in error_msg
        # Should not mention the other deps if they're available
        assert "duckduckgo-search" not in error_msg or "beautifulsoup4" not in error_msg
        
    finally:
        bfcl_v4_module.httpx = original_httpx


def test_v4_mock_mode_works_without_deps():
    """Test that mock mode works even when dependencies are missing."""
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    
    import verifiers.envs.bfcl_v4_env as bfcl_v4_module
    original_httpx = bfcl_v4_module.httpx
    original_ddgs = bfcl_v4_module.DDGS
    original_bs4 = bfcl_v4_module.BeautifulSoup
    
    try:
        # Simulate missing dependencies
        bfcl_v4_module.httpx = None
        bfcl_v4_module.DDGS = None
        bfcl_v4_module.BeautifulSoup = None
        
        # Mock mode should work fine
        env = BFCLV4WebEnv(dataset=ds, live=False)
        assert env is not None
        assert len(env.tools) == 2  # Should have search and fetch tools
        
    finally:
        # Restore original values
        bfcl_v4_module.httpx = original_httpx
        bfcl_v4_module.DDGS = original_ddgs
        bfcl_v4_module.BeautifulSoup = original_bs4

