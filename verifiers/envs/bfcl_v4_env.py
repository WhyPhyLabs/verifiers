from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from datasets import Dataset

from verifiers import ToolEnv, SingleTurnEnv, Rubric, Parser
from verifiers.types import Messages


@dataclass
class V4Config:
    include_snippets: bool = True
    fetch_fail_rate: float = 0.0
    failure_seed: int = 1234
    offline_cache_dir: Optional[str] = None
    max_turns: int = 8


def _read_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _to_dataset_v4(items: list[dict]) -> Dataset:
    prompts: list[list[dict]] = []
    answers: list[str] = []
    infos: list[dict] = []
    for it in items:
        if "prompt" in it:
            prompts.append(it["prompt"])
        else:
            q = it.get("question") or ""
            prompts.append([{"role": "user", "content": q}])
        answers.append(it.get("answer", ""))
        info = dict(it.get("info", {}))
        if "sources" in it:
            info["sources"] = it["sources"]
        if "gold_urls" in it:
            info["gold_urls"] = it["gold_urls"]
        infos.append(info)
    return Dataset.from_dict({"prompt": prompts, "answer": answers, "info": infos})


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\,\.\/\-\_\*\^\(\)]", "", s)
    s = s.strip()
    return s


def answer_exact_match(prompt: Messages, completion: Messages, answer: str, **kwargs) -> float:
    if not isinstance(completion, list) or not completion:
        return 0.0
    last = completion[-1]
    content = last.get("content", "")
    try:
        obj = json.loads(content) if isinstance(content, str) else {}
    except Exception:
        return 0.0
    if not isinstance(obj, dict) or "answer" not in obj:
        return 0.0
    pred = _normalize_answer(str(obj.get("answer", "")))
    gold = _normalize_answer(str(answer))
    return 1.0 if pred == gold else 0.0


class BFCLV4Rubric(Rubric):
    def __init__(self):
        super().__init__(funcs=[answer_exact_match], weights=[1.0], parser=Parser())


class _Cache:
    def __init__(self, root: Optional[str]):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, name: str) -> Optional[Path]:
        if not self.root:
            return None
        return self.root / name

    def get_json(self, name: str) -> Optional[Any]:
        p = self._p(name)
        if not p or not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put_json(self, name: str, obj: Any) -> None:
        p = self._p(name)
        if not p:
            return
        p.write_text(json.dumps(obj), encoding="utf-8")

    def get_text(self, name: str) -> Optional[str]:
        p = self._p(name)
        if not p or not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def put_text(self, name: str, s: str) -> None:
        p = self._p(name)
        if not p:
            return
        p.write_text(s, encoding="utf-8")


def _mk_v4_tools(cfg: V4Config) -> list[Callable]:
    cache = _Cache(cfg.offline_cache_dir)
    rng = random.Random(cfg.failure_seed)

    def duckduckgo_search(keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
        """Return [{'title': str, 'url': str, 'snippet'?: str}]"""
        key = f"search::{region}::{max_results}::{keywords}"
        if cache.root:
            obj = cache.get_json(key)
            if obj is not None:
                return obj
            if cfg.offline_cache_dir is not None:
                return []
        result = [{"title": f"About {keywords}", "url": f"https://example.com/{keywords.replace(' ', '_')}"}]
        if cfg.include_snippets:
            result[0]["snippet"] = f"Snippet for {keywords}"
        if cache.root:
            cache.put_json(key, result)
        return result[:max_results]

    def fetch_url_content(url: str, mode: Literal["raw", "markdown", "truncate"] = "raw") -> str:
        """Return page content; simulate failures via seeded RNG; cache results."""
        key = f"page::{mode}::{url}"
        if cfg.fetch_fail_rate > 0:
            u = rng.random()
            if u < cfg.fetch_fail_rate:
                failures = [
                    "ERROR: 503 Service Unavailable",
                    "ERROR: 429 Too Many Requests",
                    "ERROR: 403 Forbidden",
                    "ERROR: ConnectTimeout",
                    "ERROR: ReadTimeout",
                    "ERROR: ConnectionError",
                ]
                return failures[int(u * len(failures)) % len(failures)]
        if cache.root:
            txt = cache.get_text(key)
            if txt is not None:
                return txt
            if cfg.offline_cache_dir is not None:
                return ""
        body = f"Content fetched from {url}\n"
        if mode == "markdown":
            body = f"# {url}\n\n{body}"
        elif mode == "truncate":
            body = body[:32]
        if cache.root:
            cache.put_text(key, body)
        return body

    return [duckduckgo_search, fetch_url_content]


class BFCLV4WebEnv(ToolEnv):
    """Multi-turn web-search ToolEnv with exact tool signatures and failure injection."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        include_snippets: bool = True,
        fetch_fail_rate: float = 0.0,
        failure_seed: int = 1234,
        offline_cache_dir: Optional[str] = None,
        max_turns: int = 8,
        **kwargs: Any,
    ):
        cfg = V4Config(
            include_snippets=include_snippets,
            fetch_fail_rate=fetch_fail_rate,
            failure_seed=failure_seed,
            offline_cache_dir=offline_cache_dir,
            max_turns=max_turns,
        )
        tools = _mk_v4_tools(cfg)
        super().__init__(
            tools=tools,
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV4Rubric(),
            **kwargs,
        )


class BFCLV4SingleTurnEnv(ToolEnv):
    """Single-turn (B1): ToolEnv with max_turns=1; parallel tool calls allowed."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        include_snippets: bool = True,
        fetch_fail_rate: float = 0.0,
        failure_seed: int = 1234,
        offline_cache_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        cfg = V4Config(
            include_snippets=include_snippets,
            fetch_fail_rate=fetch_fail_rate,
            failure_seed=failure_seed,
            offline_cache_dir=offline_cache_dir,
            max_turns=1,
        )
        tools = _mk_v4_tools(cfg)
        super().__init__(
            tools=tools,
            max_turns=1,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV4Rubric(),
            **kwargs,
        )


class BFCLV4OracleSingleTurnEnv(SingleTurnEnv):
    """Single-turn (B2): oracle evidence; no tools."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV4Rubric(),
            **kwargs,
        )


def load_environment(
    version: Literal["v4"] = "v4",
    mode: Literal["multi", "single"] = "multi",
    single_turn_variant: Literal["b1", "b2"] = "b1",
    dataset_file: str | None = None,
    **kwargs: Any,
):
    ds = None
    if dataset_file:
        items = _read_jsonl(dataset_file)
        ds = _to_dataset_v4(items)
    if mode == "multi":
        return BFCLV4WebEnv(dataset=ds, **kwargs)
    if mode == "single":
        if single_turn_variant == "b1":
            return BFCLV4SingleTurnEnv(dataset=ds, **kwargs)
        if single_turn_variant == "b2":
            return BFCLV4OracleSingleTurnEnv(dataset=ds, **kwargs)
        raise ValueError(f"Unsupported v4 single_turn_variant: {single_turn_variant}")
    raise ValueError(f"Unsupported v4 mode: {mode}")
