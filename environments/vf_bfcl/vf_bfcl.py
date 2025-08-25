from __future__ import annotations

from typing import Any, Literal

from verifiers.envs.bfcl_v3_env import load_environment as load_v3
from verifiers.envs.bfcl_v4_env import load_environment as load_v4


def load_environment(
    version: Literal["v3", "v4"] = "v4",
    **kwargs: Any,
):
    if version == "v3":
        return load_v3(version="v3", **kwargs)
    if version == "v4":
        return load_v4(version="v4", **kwargs)
    raise ValueError(f"Unsupported BFCL version: {version}")
