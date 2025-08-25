from __future__ import annotations

import json

from verifiers.envs import bfcl_v3_env as v3
from verifiers.envs import bfcl_v4_env as v4


def test_v3_jsonl_loader(tmp_path):
    sample = {"question": "hello", "info": {"expected": {"tool": "set_kv", "args": {"key": "k", "value": "v"}}}}
    p = tmp_path / "v3.jsonl"
    p.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    ds = v3._to_dataset_v3(v3._read_jsonl(p))
    assert set(ds.column_names) == {"prompt", "info"}
    assert isinstance(ds[0]["prompt"], list) and isinstance(ds[0]["info"], dict)


def test_v4_jsonl_loader(tmp_path):
    sample = {"question": "world", "answer": "42", "info": {"sources": ["u"]}}
    p = tmp_path / "v4.jsonl"
    p.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    ds = v4._to_dataset_v4(v4._read_jsonl(p))
    assert set(ds.column_names) == {"prompt", "answer", "info"}
    assert isinstance(ds[0]["prompt"], list) and isinstance(ds[0]["answer"], str)
