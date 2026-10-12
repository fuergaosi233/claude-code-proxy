"""Reasoning-effort routing layer.

Claude Code 2.1.x sends effort as one of ``low | medium | high | xhigh``, but
each OpenAI-compatible upstream speaks its own envelope and vocabulary:

* NVIDIA NIM GLM-5.x: top-level ``reasoning_effort`` string ``low|high|max``.
* gpt-oss / o-series / kimi-k3 / deepseek-r1: top-level ``reasoning_effort``
  string ``low|medium|high``.
* deepseek-v4.1-flash: top-level ``reasoning_effort`` string ``low|high``
  (``medium`` rejected in-stream; nudged up to ``high``).
* NVIDIA Nemotron families: no ``reasoning_effort`` at all — reasoning is
  toggled via ``chat_template_kwargs: {enable_thinking, medium_effort}``.
* Always-on thinking models (kimi-k2-thinking, qwen3-*-thinking) and plain
  non-reasoning models: no reasoning parameter; omit the field entirely.

To keep the rest of the proxy provider-agnostic, this module normalizes any
incoming effort onto a single 4-level canonical scale and then renders the
provider-specific request fields at the boundary.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Canonical proxy-side scale, ordered: low < medium < high < max.
CANONICAL_EFFORTS = ("low", "medium", "high", "max")

DEFAULT_EFFORT = "high"


class EffortSpec:
    """The resolved reasoning configuration to apply to an OpenAI request.

    Exactly one envelope is populated:
      * ``reasoning_effort`` — a top-level string or numeric ``reasoning_effort``
        field for the request body.
      * ``chat_template_kwargs`` — a dict to merge into ``extra_body`` for
        models (Nemotron) that control thinking via chat template kwargs.
      * neither — the request should carry no reasoning parameter.
    """

    __slots__ = ("reasoning_effort", "chat_template_kwargs")

    def __init__(
        self,
        reasoning_effort: Any = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.reasoning_effort = reasoning_effort
        self.chat_template_kwargs = chat_template_kwargs

    @property
    def has_effect(self) -> bool:
        return self.reasoning_effort is not None or bool(self.chat_template_kwargs)

    def __repr__(self):  # pragma: no cover - debugging aid
        return (
            f"EffortSpec(reasoning_effort={self.reasoning_effort!r}, "
            f"chat_template_kwargs={self.chat_template_kwargs!r})"
        )


def normalize_effort(claude_value: Optional[str], config_override: Optional[str] = None) -> str:
    """Resolve the canonical effort level for a request.

    Precedence: the value Claude Code sent (from ``/effort``) > the
    REASONING_EFFORT env default > the built-in default (``high``).

    The env var is intentionally a *default*, not a hard override: in-session
    ``/effort`` always wins so the level can be changed on the fly without
    restarting the proxy.

    Claude Code's ``xhigh`` maps onto ``max`` here so the per-provider tables
    only ever see canonical values.
    """
    raw = (claude_value or config_override or "").strip().lower()
    if not raw:
        return DEFAULT_EFFORT

    if raw == "xhigh":
        return "max"
    if raw in CANONICAL_EFFORTS:
        return raw

    # Unknown value (bad env override, or a new Claude Code level we don't map
    # yet). Fall back to the default rather than failing the request.
    logger.warning("Unrecognized reasoning effort %r; falling back to %r", raw, DEFAULT_EFFORT)
    return DEFAULT_EFFORT


# deepseek-v4.1-flash originally accepted a numeric reasoning_effort (1-100),
# but the NIM endpoint changed contract on 2026-10-11 to expect a string or
# map (numeric values are rejected with "invalid type: integer ... expected
# string or map"). Live-verified string vocabulary: "high" and "low" are
# accepted; "medium" is rejected in-stream with a 400.


def resolve_effort(model_name: str, canonical: str) -> EffortSpec:
    """Translate a canonical effort level into the upstream's native envelope.

    Returns an ``EffortSpec`` describing exactly what to attach to the request,
    or an empty spec (``has_effect`` False) when the model should receive no
    reasoning parameter at all.
    """
    name = (model_name or "").lower()

    # Always-on thinking models: reasoning is forced by the chat template and
    # these endpoints reject a reasoning_effort param. Omit it entirely.
    if "-thinking" in name:
        return EffortSpec()

    # NVIDIA Nemotron families: no reasoning_effort. Thinking toggled via chat
    # template kwargs. nemotron-ultra additionally has a medium_effort mode;
    # other Nemotron variants get plain on/off.
    if "nemotron" in name:
        kwargs: Dict[str, Any] = {"enable_thinking": canonical != "low"}
        if canonical in ("medium", "high") and "ultra" in name:
            kwargs["medium_effort"] = True
        return EffortSpec(chat_template_kwargs=kwargs)

    # deepseek-v4.1-flash: string reasoning_effort, top-level (the endpoint
    # moved off the numeric 1-100 scale on 2026-10-11; integers now 400 with
    # "expected string or map"). Live-verified vocabulary: "medium" is
    # rejected in-stream, so nudge it up to high. Accepts temperature.
    if "deepseek-v4" in name:
        if canonical in ("medium", "max"):
            return EffortSpec(reasoning_effort="high")
        return EffortSpec(reasoning_effort=canonical)

    # NVIDIA NIM GLM-5.x family: low|high|max, no "medium". Nudge medium up to
    # high — for a coding assistant a weak answer is worse than a slower one.
    if "glm-5" in name or name.startswith("z-ai") or "z-ai/" in name:
        if canonical == "medium":
            return EffortSpec(reasoning_effort="high")
        return EffortSpec(reasoning_effort=canonical)  # low/high/max map 1:1

    # gpt-oss: low|medium|high, cap max -> high.
    if "gpt-oss" in name:
        return EffortSpec(reasoning_effort="high" if canonical == "max" else canonical)

    # Everything else matched as a reasoning model (deepseek-r1, kimi-k3,
    # o-series, ...): OpenAI-style vocabulary, cap max -> high.
    return EffortSpec(reasoning_effort="high" if canonical == "max" else canonical)


def provider_effort(model_name: str, canonical: str) -> Optional[Any]:
    """Backwards-compatible helper: just the top-level reasoning_effort value.

    Only meaningful for the string/numeric envelope; returns ``None`` for
    template-kwargs or always-on models. New code should use ``resolve_effort``.
    """
    return resolve_effort(model_name, canonical).reasoning_effort
