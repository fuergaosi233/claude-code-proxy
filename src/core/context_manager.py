"""
Context window limits and output token caps for common models.

These are conservative defaults. Override via env vars:
  MODEL_CONTEXT_LIMITS=model_name:context_tokens:max_output,...
  CONTEXT_TRIM_RATIO=0.85   (fraction of context to keep after trimming, default 0.85)
"""

from __future__ import annotations

import logging
import os
import re

_log = logging.getLogger("context_manager")

# ---------------------------------------------------------------------------
# Built-in context limits: model_name_substring -> (context_window, max_output)
# Keys are lowercased; first match wins.  These are conservative values.
# ---------------------------------------------------------------------------
_BUILTIN_LIMITS: list[tuple[str, int, int]] = [
    # OpenAI
    ("gpt-4o", 128_000, 16_384),
    ("gpt-4-turbo", 128_000, 4_096),
    ("gpt-4", 8_192, 4_096),
    ("gpt-3.5-turbo", 16_385, 4_096),
    ("o1", 200_000, 100_000),
    ("o3", 200_000, 100_000),
    ("o4", 200_000, 100_000),
    # Anthropic (if proxied through)
    ("claude-3-7", 200_000, 128_000),
    ("claude-3-5", 200_000, 8_192),
    ("claude-3-opus", 200_000, 4_096),
    ("claude-3", 200_000, 4_096),
    # DeepSeek
    ("deepseek-r1", 64_000, 8_192),
    ("deepseek", 64_000, 8_192),
    # NVIDIA NIM & Configured Models
    ("glm-5.3-flash", 1_048_576, 128_000),
    ("glm-5.3", 1_048_576, 128_000),
    ("nemotron-3-ultra-550b", 1_048_576, 32_768),
    ("nemotron-3.5-lightning-30b", 1_048_576, 32_768),
    ("kimi-k3", 1_048_576, 32_768),
    ("deepseek-v4.1-flash", 1_048_576, 32_768),
    ("laguna-xs-2.1", 262_144, 32_000),
    ("muse-glimmer-30b", 131_072, 16_384),
    ("mistral-nemotron", 128_000, 16_384),
    ("gpt-oss-20b", 131_072, 16_384),
    ("gpt-oss", 131_072, 16_384),
    ("nv-", 128_000, 4_096),
]


# Conservative estimate: characters per token (English text ~ 4 chars/token)
_CHARS_PER_TOKEN = 4


def _parse_env_overrides() -> dict[str, tuple[int, int]]:
    """Parse MODEL_CONTEXT_LIMITS env var into a dict."""
    raw = os.environ.get("MODEL_CONTEXT_LIMITS", "")
    out: dict[str, tuple[int, int]] = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            name, ctx, out_tok = parts
            try:
                out[name.strip().lower()] = (int(ctx), int(out_tok))
            except ValueError:
                pass
    return out


_ENV_OVERRIDES = _parse_env_overrides()

# Limits discovered at runtime by querying the upstream /v1/models endpoint.
# Populated by provider_discovery.run_discovery_and_patch() on startup.
# Keyed by exact model ID as returned by the provider.
_DISCOVERED_LIMITS: dict[str, tuple[int, int]] = {}

# Fraction of context window chars to keep when trimming (leaves headroom for output)
_TRIM_RATIO = float(os.environ.get("CONTEXT_TRIM_RATIO", "0.85"))


def get_model_limits(model_name: str) -> tuple[int, int]:
    """Return (context_window_tokens, max_output_tokens) for a model.

    Lookup priority (highest → lowest):
      1. Env overrides  (MODEL_CONTEXT_LIMITS)
      2. Runtime-discovered limits from the upstream /v1/models endpoint
      3. Built-in static table
      4. Safe default (32_000, 4_096)
    """
    name_lower = (model_name or "").lower()

    # 1. env overrides (exact match first, then substring)
    if name_lower in _ENV_OVERRIDES:
        return _ENV_OVERRIDES[name_lower]
    for key, limits in _ENV_OVERRIDES.items():
        if key in name_lower:
            return limits

    # 2. runtime-discovered limits (exact match then substring)
    if model_name in _DISCOVERED_LIMITS:
        return _DISCOVERED_LIMITS[model_name]
    for disc_id, limits in _DISCOVERED_LIMITS.items():
        if disc_id.lower() in name_lower or name_lower in disc_id.lower():
            return limits

    # 3. built-in table (substring match in order)
    for marker, ctx, out in _BUILTIN_LIMITS:
        if marker in name_lower:
            return ctx, out

    # 4. safe default
    return 32_000, 4_096


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate from a list of OpenAI-format messages.

    Uses character count / 4 as a proxy — accurate enough for budget decisions.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
        # tool_call arguments
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total_chars += len(fn.get("arguments", ""))
    return total_chars // _CHARS_PER_TOKEN


def trim_messages_to_fit(
    messages: list[dict],
    model_name: str,
    requested_max_tokens: int,
) -> tuple[list[dict], int]:
    """Trim the messages list so input + output fits within the model's context window.

    Strategy (preserves correctness):
      1. Always keep the *first* message (usually carries the system prompt).
      2. Always keep the *last* message (the new user turn).
      3. Drop the oldest middle turns until we fit, keeping tool-call pairs intact.
      4. Clamp requested_max_tokens to the model's max output cap.

    Returns (trimmed_messages, clamped_max_tokens).
    """
    ctx_window, max_output = get_model_limits(model_name)
    # Budget for input: reserve space for the output
    input_budget_tokens = int(ctx_window * _TRIM_RATIO) - requested_max_tokens
    if input_budget_tokens <= 0:
        # Even the output alone is too large — clamp output to half the context
        requested_max_tokens = ctx_window // 2
        input_budget_tokens = ctx_window // 2

    # Clamp max_tokens to model's max output
    clamped_max_tokens = min(requested_max_tokens, max_output)

    estimated = estimate_tokens(messages)
    if estimated <= input_budget_tokens:
        # Fits fine — only clamp the output cap
        return messages, clamped_max_tokens

    _log.warning(
        "Context trim: model=%s estimated_input=%d tokens, budget=%d tokens "
        "(ctx_window=%d, max_output=%d). Dropping oldest turns.",
        model_name,
        estimated,
        input_budget_tokens,
        ctx_window,
        max_output,
    )

    if len(messages) <= 2:
        # Only 1-2 messages; nothing to trim — let the API decide
        _log.warning("Cannot trim: only %d messages, passing as-is.", len(messages))
        return messages, clamped_max_tokens

    # Keep first and last; trim from index 1 onwards
    first = messages[0]
    last = messages[-1]
    middle = list(messages[1:-1])

    # Drop from the oldest (front of middle) in chunks of 2 to preserve turn pairs
    while middle and estimate_tokens([first] + middle + [last]) > input_budget_tokens:
        # Drop one turn (or two to preserve assistant→tool_result pairing)
        if len(middle) >= 2:
            middle = middle[2:]
        else:
            middle = middle[1:]

    trimmed = [first] + middle + [last]
    new_estimate = estimate_tokens(trimmed)
    _log.info(
        "Context trim done: %d -> %d messages, ~%d -> ~%d tokens",
        len(messages),
        len(trimmed),
        estimated,
        new_estimate,
    )
    return trimmed, clamped_max_tokens
