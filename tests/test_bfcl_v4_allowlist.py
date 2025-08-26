from __future__ import annotations

import pytest
from datasets import Dataset

from verifiers.envs.bfcl_v4_env import BFCLV4WebEnv


def test_v4_fetch_url_allowlist_blocks_non_http():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds)
    # Simulate blocked scheme
    out = env.tool_map["fetch_url_content"](url="file:///etc/passwd", mode="raw")  # type: ignore
    assert out.startswith("ERROR:")


def test_v4_fetch_url_allowlist_blocks_private_reserved_ips():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds)
    
    # Test various private/reserved IP addresses that should be blocked
    blocked_urls = [
        # Loopback addresses
        "http://127.0.0.1",
        "http://127.0.0.1:8080",
        "http://[::1]",  # IPv6 loopback
        
        # Private network ranges (RFC 1918)
        "http://10.0.0.1",
        "http://10.255.255.254",
        "http://172.16.0.1", 
        "http://172.31.255.254",
        "http://192.168.1.1",
        "http://192.168.255.254",
        
        # Link-local addresses
        "http://169.254.169.254",  # Common metadata service
        "http://169.254.1.1",
        "http://[fe80::1]",  # IPv6 link-local
        
        # Reserved/multicast ranges
        "http://224.0.0.1",  # Multicast
        "http://240.0.0.1",  # Reserved
        "http://[ff02::1]",  # IPv6 multicast
    ]
    
    for url in blocked_urls:
        out = env.tool_map["fetch_url_content"](url=url, mode="raw")  # type: ignore
        assert out.startswith("ERROR:"), f"URL {url} should be blocked but wasn't"


def test_v4_fetch_url_allowlist_allows_public_urls():
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds)
    
    # Test public URLs that should be allowed (note: these will fail with network errors, but shouldn't be blocked by allowlist)
    allowed_urls = [
        "http://example.com",
        "https://example.com",
        "http://example.com:8080",
        "https://subdomain.example.com/path",
        "http://8.8.8.8",  # Public IP (Google DNS)
        "http://1.1.1.1",  # Public IP (Cloudflare DNS)
    ]
    
    for url in allowed_urls:
        out = env.tool_map["fetch_url_content"](url=url, mode="raw")  # type: ignore
        # These should either succeed or fail with network/timeout errors, but not allowlist errors
        assert not out.startswith("ERROR: Blocked"), f"URL {url} should not be blocked by allowlist"
