from __future__ import annotations

from typing import Any, Callable, Literal, cast

from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from bfcl_eval.model_handler.utils import decoded_output_to_execution_list
from bfcl_eval.utils import (
    is_executable_format_output,
    is_function_calling_format_output,
)

from verifiers.parsers.parser import Parser
from verifiers.types import ChatMessage, Messages

__all__ = ["BFCLParser"]

BFCLMode = Literal["ast", "execute"]


class BFCLParser(Parser):
    """Parser that delegates decoding to bfcl_eval helpers."""

    def __init__(
        self,
        mode: BFCLMode = "ast",
        *,
        postprocess: Callable[[Any], Any] | None = None,
        require_valid: bool = False,
        language: str | None = None,
    ) -> None:
        super().__init__(extract_fn=lambda value: value)
        self.mode = mode
        self._postprocess = postprocess
        self.require_valid = require_valid
        self._decode = (
            default_decode_ast_prompting
            if mode == "ast"
            else default_decode_execute_prompting
        )
        # Language hint used by bfcl_eval's AST decoder (Python/Java/JavaScript)
        self.language = language or "Python"

    def parse(self, text: str) -> Any:
        decoded = self._try_decode(text)

        if decoded is None:
            return self._fallback_value(text)

        if self._postprocess is not None:
            try:
                decoded = self._postprocess(decoded)
            except Exception as exc:  # pragma: no cover - defensive log path
                self.logger.warning("BFCLParser postprocess failed: %s", exc)
                return decoded

        return decoded

    def parse_answer(self, completion: Messages) -> Any:
        if isinstance(completion, str):
            return self.parse(completion)
        messages = [cast(ChatMessage, m) for m in completion if isinstance(m, dict)]
        assistants = self.get_assistant_messages(messages)
        for message in reversed(assistants):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return self.parse(content)
        return self._fallback_value("")

    def get_format_reward_func(self) -> Callable[..., float]:
        def format_reward(completion: Messages, **kwargs) -> float:
            messages = (
                [cast(ChatMessage, m) for m in completion if isinstance(m, dict)]
                if isinstance(completion, list)
                else [cast(ChatMessage, {"role": "assistant", "content": completion})]
            )
            assistant_messages = self.get_assistant_messages(messages)
            if not assistant_messages:
                return 0.0

            total = len(assistant_messages)
            valid = 0
            for message in assistant_messages:
                content = message.get("content")
                if not isinstance(content, str) or not content:
                    continue
                decoded = self._try_decode(content)
                if decoded is None:
                    continue
                if self.mode == "ast":
                    if is_function_calling_format_output(decoded):
                        valid += 1
                else:
                    if is_executable_format_output(decoded):
                        valid += 1
            return valid / total

        return format_reward

    # --- Multi-turn helpers ---
    @staticmethod
    def _tool_calls_to_exec_list(tool_calls: list) -> list[str]:
        """Convert OpenAI tool_calls to BFCL execution strings using BFCL helpers."""
        calls: list[dict[str, dict]] = []
        for tc in tool_calls or []:
            try:
                if hasattr(tc, "function"):
                    name = tc.function.name  # type: ignore[attr-defined]
                    raw_args = tc.function.arguments or "{}"  # type: ignore[attr-defined]
                else:
                    fn = tc.get("function", {})  # type: ignore[assignment]
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                import json as _json

                args = _json.loads(raw_args) if isinstance(raw_args, str) else {}
                calls.append({name: args})
            except Exception:
                continue
        return decoded_output_to_execution_list(calls)

    def decode_multi_turn_execute(self, messages: Messages) -> list[list[list[str]]]:
        """Decode a chat transcript into BFCL execute lists per turn/step.

        Returns list[turn][step][str] matching multi_turn_checker input.
        """
        msgs = (
            [cast(ChatMessage, m) for m in messages if isinstance(m, dict)]
            if isinstance(messages, list)
            else [cast(ChatMessage, {"role": "assistant", "content": messages})]
        )
        # Find user message indices to split turns
        user_idx = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if not user_idx:
            return []
        user_idx.append(len(msgs))
        all_turns: list[list[list[str]]] = []
        for t in range(len(user_idx) - 1):
            start = user_idx[t]
            end = user_idx[t + 1]
            steps: list[list[str]] = []
            for i in range(start + 1, end):
                m = msgs[i]
                if m.get("role") != "assistant":
                    continue
                if m.get("tool_calls"):
                    steps.append(self._tool_calls_to_exec_list(m["tool_calls"]))
                else:
                    content = m.get("content") or ""
                    decoded = default_decode_execute_prompting(str(content))
                    if isinstance(decoded, list):
                        steps.append(decoded)
                    else:
                        steps.append([])
            all_turns.append(steps)
        return all_turns

    def _try_decode(self, text: str) -> Any | None:
        if not text:
            return [] if self.mode == "ast" else []
        try:
            if self.mode == "ast":
                decoded = self._decode(text, language=self.language)
            else:
                decoded = self._decode(text)
        except Exception as exc:  # pragma: no cover - log parsing issues
            self.logger.debug("BFCLParser decode failed: %s", exc)
            return None
        return decoded

    def _fallback_value(self, text: str) -> Any:
        if self.require_valid:
            # For AST we expect a list[dict], for execute a list[str]
            return []
        return text
