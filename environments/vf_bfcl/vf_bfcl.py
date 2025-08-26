from __future__ import annotations

from typing import Any, Literal

from verifiers.envs.bfcl_v3_env import load_environment as load_v3
from verifiers.envs.bfcl_v4_env import load_environment as load_v4


def load_environment(
    version: Literal["v3", "v4", "v3_single", "v4_single", "v4_oracle"] = "v4",
    **kwargs: Any,
):
    # Handle version aliases for CLI compatibility
    if version in ("v3_single",):
        kwargs["mode"] = "single"
        return load_v3(version="v3", **kwargs)
    if version in ("v3",):
        kwargs.setdefault("mode", "multi")
        return load_v3(version="v3", **kwargs)

    if version in ("v4_single",):
        kwargs["mode"] = "single"
        kwargs.setdefault("single_turn_variant", "b1")
        return load_v4(version="v4", **kwargs)
    if version in ("v4_oracle",):
        kwargs["mode"] = "single"
        kwargs["single_turn_variant"] = "b2"
        return load_v4(version="v4", **kwargs)
    if version == "v4":
        kwargs.setdefault("mode", "multi")
        return load_v4(version="v4", **kwargs)
    raise ValueError(f"Unsupported BFCL version: {version}")
