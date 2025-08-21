from __future__ import annotations

from .rubric import Rubric
from ..types import Messages, State


class MultiSWERubric(Rubric):
    """Simple rubric assigning 1.0 if tests pass, else 0.0."""

    def __init__(self) -> None:
        super().__init__(funcs=[self.pass_reward], weights=[1.0])

    def pass_reward(self, parser, completion: Messages, answer: str | None = None, state: State | None = None) -> float:  # noqa: D401, ARG002
        return 1.0 if state and state.get("passed") else 0.0
