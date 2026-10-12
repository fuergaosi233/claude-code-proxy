"""Tests for context_manager: model limits lookup and message trimming."""

import pytest

from src.core.context_manager import (
    estimate_tokens,
    get_model_limits,
    trim_messages_to_fit,
)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


class TestGetModelLimits:
    def test_gpt4o(self):
        ctx, out = get_model_limits("gpt-4o")
        assert ctx == 128_000
        assert out == 16_384

    def test_gpt4o_with_date(self):
        ctx, out = get_model_limits("gpt-4o-2024-11-20")
        assert ctx == 128_000

    def test_gpt35(self):
        ctx, _ = get_model_limits("gpt-3.5-turbo")
        assert ctx == 16_385

    def test_deepseek_r1(self):
        ctx, _ = get_model_limits("deepseek-r1")
        assert ctx == 64_000

    def test_claude_35(self):
        ctx, out = get_model_limits("claude-3-5-sonnet-20241022")
        assert ctx == 200_000
        assert out == 8_192

    def test_unknown_model_returns_default(self):
        ctx, out = get_model_limits("my-exotic-llm-v2")
        assert ctx == 32_000
        assert out == 4_096

    def test_gpt_oss(self):
        ctx, _ = get_model_limits("openai/gpt-oss-20b")
        assert ctx in (128_000, 131_072)


class TestEstimateTokens:
    def test_plain_string_messages(self):
        msgs = [_msg("user", "A" * 400)]  # 400 chars -> ~100 tokens
        assert estimate_tokens(msgs) == 100

    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "A" * 800}]}]
        assert estimate_tokens(msgs) == 200


class TestTrimMessagesToFit:
    def _make_long_conversation(self, n_turns: int, chars_each: int = 100) -> list[dict]:
        """Make a conversation with n_turns user/assistant pairs."""
        msgs = [_msg("user", "System context. " * 5)]
        for i in range(n_turns):
            msgs.append(_msg("user", f"Turn {i}: " + "x" * chars_each))
            msgs.append(_msg("assistant", f"Reply {i}: " + "y" * chars_each))
        msgs.append(_msg("user", "Final question? " + "z" * chars_each))
        return msgs

    def test_short_conversation_passes_through(self):
        msgs = [_msg("user", "Hello"), _msg("assistant", "Hi"), _msg("user", "Bye")]
        trimmed, max_tok = trim_messages_to_fit(msgs, "gpt-4o", 512)
        assert trimmed == msgs
        assert max_tok == 512

    def test_max_tokens_clamped_to_model_output_cap(self):
        msgs = [_msg("user", "x")]
        _, max_tok = trim_messages_to_fit(msgs, "gpt-3.5-turbo", 99_999)
        assert max_tok == 4_096  # model cap

    def test_very_long_conversation_gets_trimmed(self):
        # 200 turns with 400 chars each -> ~20k tokens, well within gpt-4o
        # but with a tiny artificial limit we can force trimming
        msgs = self._make_long_conversation(n_turns=50, chars_each=800)
        before = len(msgs)
        # Use a model with a small context by passing a custom name via env
        # Instead, test via a very small fake budget by monkeypatching
        import src.core.context_manager as cm

        orig = cm._BUILTIN_LIMITS[:]
        try:
            cm._BUILTIN_LIMITS.insert(0, ("test-tiny", 2_000, 512))
            trimmed, _ = trim_messages_to_fit(msgs, "test-tiny", 256)
            assert len(trimmed) < before
        finally:
            cm._BUILTIN_LIMITS[:] = orig

    def test_first_and_last_message_preserved(self):
        msgs = self._make_long_conversation(n_turns=50, chars_each=800)
        import src.core.context_manager as cm

        orig = cm._BUILTIN_LIMITS[:]
        try:
            cm._BUILTIN_LIMITS.insert(0, ("test-tiny2", 2_000, 512))
            trimmed, _ = trim_messages_to_fit(msgs, "test-tiny2", 256)
            assert trimmed[0] == msgs[0]
            assert trimmed[-1] == msgs[-1]
        finally:
            cm._BUILTIN_LIMITS[:] = orig
