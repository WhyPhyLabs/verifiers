from __future__ import annotations

import ipaddress
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from datasets import Dataset

from verifiers import ToolEnv, SingleTurnEnv, Rubric, Parser
from verifiers.types import Messages

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dep
    httpx = None  # type: ignore

try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dep
    DDGS = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dep
    BeautifulSoup = None  # type: ignore


@dataclass
class V4Config:
    include_snippets: bool = True
    fetch_fail_rate: float = 0.0
    failure_seed: int = 1234
    offline_cache_dir: Optional[str] = None
    max_turns: int = 8
    live: bool = False  # live web mode for search/fetch
    timeout_s: float = 8.0
    max_retries: int = 2


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

        answer = it.get("answer", "")
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0]
        answers.append(answer)

        info = it.get("info", {})
        if isinstance(info, list) and len(info) > 0:
            info = info[0]
        elif isinstance(info, list):
            info = {}
        info = dict(info)
        if "sources" in it:
            info["sources"] = it["sources"]
        if "gold_urls" in it:
            info["gold_urls"] = it["gold_urls"]
        if "evidence" in it:
            info["evidence"] = it["evidence"]
        infos.append(info)
    
    return Dataset.from_dict({"prompt": prompts, "answer": answers, "info": infos})


def _normalize_answer(s: str) -> str:
    import unicodedata
    
    def remove_diacritics(text: str) -> str:
        # Normalize to NFD (decomposed form) first
        text = unicodedata.normalize("NFD", text)
        # Remove combining diacritical marks
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        # Normalize back to NFC
        return unicodedata.normalize("NFC", text)
    
    # Remove diacritics first
    s = remove_diacritics(s)
    
    # Unicode normalization (NFKC) - compatibility decomposition followed by canonical composition
    s = unicodedata.normalize("NFKC", s)
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove punctuation and special characters (including em dash and other Unicode punctuation)
    s = re.sub(r"[\,\.\/\-\_\*\^\(\)\[\]\{\}\:\;\!\?\>\<\@\#\$\%\^\&\+\=\~\`\'\"\|\\—–]", "", s)
    
    # Normalize whitespace - replace multiple spaces with single space and strip
    s = re.sub(r"\s+", " ", s)
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

    def _sanitize_filename(self, name: str) -> str:
        # Replace characters that are problematic in file paths
        # Replace slashes and colons with underscores
        sanitized = name.replace('/', '_').replace(':', '_')
        # Replace other problematic characters
        sanitized = sanitized.replace('\\', '_').replace('?', '_').replace('*', '_')
        return sanitized

    def _p(self, name: str) -> Optional[Path]:
        if not self.root:
            return None
        sanitized_name = self._sanitize_filename(name)
        return self.root / sanitized_name

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


# --- Modular Search API ---
class SearchClient:
    def search(self, keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:  # pragma: no cover - interface
        raise NotImplementedError


class MockSearchClient(SearchClient):
    def __init__(self, include_snippets: bool = True):
        self.include_snippets = include_snippets

    def search(self, keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
        res = [{"title": f"About {keywords}", "url": f"https://example.com/{keywords.replace(' ', '_')}"}]
        if self.include_snippets:
            res[0]["snippet"] = f"Snippet for {keywords}"
        return res[:max_results]


class DDGSearchClient(SearchClient):
    def __init__(self, include_snippets: bool = True):
        self.include_snippets = include_snippets

    def search(self, keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
        if DDGS is None:
            return []
        out: list[dict] = []
        with DDGS() as ddgs:  # type: ignore
            for r in ddgs.text(keywords, region=region, max_results=max_results):  # type: ignore
                item = {"title": r.get("title", ""), "url": r.get("href", "")}
                if self.include_snippets and r.get("body"):
                    item["snippet"] = r.get("body")
                out.append(item)
        return out[:max_results]


# --- Fetch (live or mock) with allowlist & retries ---
_ALLOWED_SCHEMES = {"http", "https"}


def _is_allowed_url(url: str) -> bool:
    try:
        if not any(url.lower().startswith(s + "://") for s in _ALLOWED_SCHEMES):
            return False
        # Disallow IP literals that are private/reserved
        host = url.split("://", 1)[1].split("/", 1)[0]
        # strip port
        host = host.split(":")[0]
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast:
                return False
        except ValueError:
            # not an IP; perform DNS resolution to check for rebind attacks
            try:
                import socket
                # Resolve hostname to IP addresses
                addr_info = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                for family, _, _, _, sockaddr in addr_info:
                    ip = sockaddr[0] if isinstance(sockaddr, tuple) else sockaddr
                    try:
                        ip_obj = ipaddress.ip_address(ip)
                        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local or ip_obj.is_multicast:
                            return False
                    except ValueError:
                        continue
            except Exception:
                # If DNS resolution fails, be conservative and block
                return False
        return True
    except Exception:
        return False


@dataclass
class Fetcher:
    timeout_s: float
    max_retries: int
    live: bool
    max_redirects: int = 3

    def fetch(self, url: str, mode: Literal["raw", "markdown", "truncate"] = "raw") -> str:
        if not _is_allowed_url(url):
            return "ERROR: URL not allowed"
        if not self.live or httpx is None:
            body = f"Content fetched from {url}\n"
            return self._postprocess(body, mode)
        
        # Handle redirects manually with security checks
        current_url = url
        redirect_count = 0
        
        delay = 0.2
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s, follow_redirects=False) as client:  # type: ignore
                    resp = client.get(current_url)
                    
                    # Handle redirect
                    if resp.status_code in (301, 302, 303, 307, 308):
                        redirect_count += 1
                        if redirect_count > self.max_redirects:
                            return "ERROR: too many redirects"
                        
                        # Get redirect location
                        location = resp.headers.get("location")
                        if not location:
                            return "ERROR: redirect with no Location header"
                        
                        # Handle relative redirects
                        if location.startswith("/"):
                            from urllib.parse import urlparse
                            parsed = urlparse(current_url)
                            location = f"{parsed.scheme}://{parsed.netloc}{location}"
                        
                        # Security check on redirect target
                        if not _is_allowed_url(location):
                            return "ERROR: redirect to disallowed URL"
                        
                        current_url = location
                        continue  # Try again with the new URL
                    
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    
                    text = resp.text
                    return self._postprocess(text, mode)
            except Exception:
                if attempt >= self.max_retries:
                    return "ERROR: fetch failed"
                time.sleep(delay)
                delay *= 2
        return "ERROR: fetch failed"

    def _postprocess(self, text: str, mode: str) -> str:
        if mode == "truncate":
            return text[:256]
        if mode == "markdown":
            # best-effort HTML→text if BeautifulSoup available
            if BeautifulSoup is not None and ("<html" in text.lower() or "<p" in text.lower()):
                try:
                    soup = BeautifulSoup(text, "html.parser")  # type: ignore
                    md = soup.get_text("\n")
                    return md
                except Exception:
                    pass
        return text


def _mk_v4_tools(cfg: V4Config, search_client: Optional[SearchClient] = None) -> list[Callable]:
    cache = _Cache(cfg.offline_cache_dir)
    rng = random.Random(cfg.failure_seed)
    search_client = search_client or (DDGSearchClient(cfg.include_snippets) if cfg.live else MockSearchClient(cfg.include_snippets))
    fetcher = Fetcher(timeout_s=cfg.timeout_s, max_retries=cfg.max_retries, live=cfg.live)

    def duckduckgo_search(keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict]:
        key = f"search::{region}::{max_results}::{keywords}"
        if cache.root:
            obj = cache.get_json(key)
            if obj is not None:
                return obj
            if cfg.offline_cache_dir is not None and not cfg.live:
                return []
        result = search_client.search(keywords, max_results=max_results, region=region)
        if cache.root:
            cache.put_json(key, result)
        return result[:max_results]

    def fetch_url_content(url: str, mode: Literal["raw", "markdown", "truncate"] = "raw") -> str:
        key = f"page::{mode}::{url}"
        # Failure injection
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
            if cfg.offline_cache_dir is not None and not cfg.live:
                return ""
        body = fetcher.fetch(url, mode)
        if cache.root and body and not body.startswith("ERROR:"):
            cache.put_text(key, body)
        return body

    return [duckduckgo_search, fetch_url_content]


class BFCLV4WebEnv(ToolEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        include_snippets: bool = True,
        fetch_fail_rate: float = 0.0,
        failure_seed: int = 1234,
        offline_cache_dir: Optional[str] = None,
        max_turns: int = 8,
        live: bool = False,
        search_client: Optional[SearchClient] = None,
        timeout_s: float = 8.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        # Guard for live mode dependencies
        if live:
            missing_deps = []
            if httpx is None:
                missing_deps.append("httpx")
            if DDGS is None:
                missing_deps.append("duckduckgo-search")
            if BeautifulSoup is None:
                missing_deps.append("beautifulsoup4")
            
            if missing_deps:
                raise ImportError(
                    f"Live mode requires additional dependencies. Please install with: "
                    f"pip install 'verifiers[bfcl]'\n"
                    f"Missing dependencies: {', '.join(missing_deps)}"
                )
        
        cfg = V4Config(
            include_snippets=include_snippets,
            fetch_fail_rate=fetch_fail_rate,
            failure_seed=failure_seed,
            offline_cache_dir=offline_cache_dir,
            max_turns=max_turns,
            live=live,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        tools = _mk_v4_tools(cfg, search_client=search_client)
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
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        include_snippets: bool = True,
        fetch_fail_rate: float = 0.0,
        failure_seed: int = 1234,
        offline_cache_dir: Optional[str] = None,
        live: bool = False,
        search_client: Optional[SearchClient] = None,
        timeout_s: float = 8.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        cfg = V4Config(
            include_snippets=include_snippets,
            fetch_fail_rate=fetch_fail_rate,
            failure_seed=failure_seed,
            offline_cache_dir=offline_cache_dir,
            max_turns=1,
            live=live,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        tools = _mk_v4_tools(cfg, search_client=search_client)
        super().__init__(
            tools=tools,
            max_turns=1,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV4Rubric(),
            **kwargs,
        )
        # Force max_turns to 1 for single-turn environment
        self.max_turns = 1


def _inject_oracle(ds: Dataset) -> Dataset:
    def add_evidence(prompt: list[dict], info: dict) -> list[dict]:
        ev = info.get("evidence")
        if not ev:
            return prompt
        sys_msg = {"role": "system", "content": f"### Evidence (oracle)\n{ev}"}
        # Handle the case where prompt might be nested
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], list):
            # prompt is a list containing a list, flatten it
            return [sys_msg] + prompt[0]
        else:
            # prompt is already a list of dicts
            return [sys_msg] + prompt

    prompts = []
    answers = []
    infos = []
    for i in range(len(ds)):
        prompt = add_evidence(ds[i]["prompt"], ds[i]["info"])
        # Ensure prompt is always a list of dicts
        if isinstance(prompt, list):
            prompts.append(prompt)
        else:
            prompts.append([prompt])
        answers.append(ds[i]["answer"])
        infos.append(ds[i]["info"])
    
    return Dataset.from_dict({"prompt": prompts, "answer": answers, "info": infos})


class BFCLV4OracleSingleTurnEnv(SingleTurnEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        inject_oracle: bool = True,
        **kwargs: Any,
    ):
        if dataset is not None and inject_oracle:
            dataset = _inject_oracle(dataset)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=Parser(),
            rubric=BFCLV4Rubric(),
            max_turns=1,
            **kwargs,
        )
        # Force max_turns to 1 for single-turn environment
        self.max_turns = 1


def load_bfcl_v4(path: str | Path) -> Dataset:
    """Load BFCL v4 dataset from JSONL file."""
    items = _read_jsonl(path)
    return _to_dataset_v4(items)


def load_environment(
    version: Literal["v4"] = "v4",
    mode: Literal["multi", "single"] = "multi",
    single_turn_variant: Literal["b1", "b2"] = "b1",
    dataset_file: str | None = None,
    dataset: Dataset | None = None,
    **kwargs: Any,
):
    ds = dataset
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
