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
        "http://[fe80::2:3:4:5:6:7:8]",  # IPv6 link-local with full address
        
        # Reserved/multicast ranges
        "http://224.0.0.1",  # Multicast
        "http://240.0.0.1",  # Reserved
        "http://[ff02::1]",  # IPv6 multicast
        "http://[ff02::1:2:3:4:5:6:7]",  # IPv6 multicast with full address
        
        # IPv6 Unique Local Addresses (ULA) - RFC 4193 (private IPv6 equivalent)
        "http://[fc00::1]",  # ULA prefix
        "http://[fd00::1]",  # ULA prefix (locally assigned)
        "http://[fd12:3456:789a:bcde::1]",  # Full ULA address
        "http://[fd12:3456:789a:bcde:f012:3456:789a:bcde]",  # Full ULA address
        
        # IPv6 documentation prefixes (RFC 3849)
        "http://[2001:db8::1]",  # Documentation prefix
        
        # IPv6 benchmarking (RFC 5180)
        "http://[2001:2::1]",  # Benchmarking
        
        # IPv6 6to4 relay anycast (RFC 3068)
        "http://[192:88:99::1]",  # 6to4 relay
        
        # IPv6 teredo tunneling (RFC 4380)
        "http://[2001::::1]",  # Teredo prefix
        
        # IPv6 ORCHIDv2 (RFC 7343)
        "http://[2001:10::1]",  # ORCHIDv2
    ]
    
    for url in blocked_urls:
        out = env.tool_map["fetch_url_content"](url=url, mode="raw")  # type: ignore
        assert out.startswith("ERROR:"), f"URL {url} should be blocked but wasn't"


def test_v4_fetch_url_allowlist_blocks_ipv6_private_ula_comprehensive():
    """Comprehensive test for IPv6 private and ULA (Unique Local Address) ranges."""
    ds = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["a"], "info": [{}]})
    env = BFCLV4WebEnv(dataset=ds)
    
    # Test IPv6 private/ULA ranges that should be blocked
    ipv6_private_urls = [
        # Unique Local Addresses (ULA) - fc00::/7
        "http://[fc00::1]",
        "http://[fc00::ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
        "http://[fd00::1]",
        "http://[fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
        "http://[fd12:3456:789a::1]",
        "http://[fd12:3456:789a:bcde::1]",
        "http://[fd12:3456:789a:bcde:f012:3456:789a:bcde]",
        
        # Link-local addresses - fe80::/10
        "http://[fe80::1]",
        "http://[fe80::ffff:ffff:ffff:ffff]",
        "http://[febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
        "http://[fe80::1:2:3:4:5:6:7]",
        
        # Site-local addresses (deprecated but should still be blocked) - fec0::/10
        "http://[fec0::1]",
        "http://[fec0::ffff:ffff:ffff:ffff]",
        "http://[fecf:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
        
        # Documentation prefix - 2001:db8::/32
        "http://[2001:db8::1]",
        "http://[2001:db8::ffff:ffff:ffff:ffff:ffff:ffff]",
        "http://[2001:db8:1:2:3:4:5:6:7]",
        
        # Benchmarking - 2001:2::/28
        "http://[2001:2::1]",
        "http://[2001:2:ffff::1]",
        "http://[2001:2:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
        
        # 6to4 relay anycast - 192:88:99::/40
        "http://[192:88:99::1]",
        "http://[192:88:99:ffff:ffff:ffff:ffff:ffff:ffff]",
        
        # Teredo tunneling - 2001::/32
        "http://[2001::1]",
        "http://[2001::ffff:ffff:ffff:ffff:ffff:ffff]",
        "http://[2001:0:1:2:3:4:5:6:7]",
        
        # ORCHIDv2 - 2001:10::/28
        "http://[2001:10::1]",
        "http://[2001:10:ffff::1]",
        "http://[2001:10:ffff:ffff:ffff:ffff:ffff:ffff:ffff]",
    ]
    
    for url in ipv6_private_urls:
        out = env.tool_map["fetch_url_content"](url=url, mode="raw")  # type: ignore
        assert out.startswith("ERROR:"), f"IPv6 URL {url} should be blocked but wasn't"


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
