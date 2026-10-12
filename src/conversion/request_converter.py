import json
import logging
from typing import Any, Dict, List

from src.conversion.effort_mapper import (
    normalize_effort,
    provider_effort,
    resolve_effort,
)
from src.core.config import config
from src.core.constants import Constants
from src.core.context_manager import trim_messages_to_fit
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest

logger = logging.getLogger(__name__)


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Dict[str, Any]:
    """Convert Claude API request format to OpenAI format."""

    # Map model. map_claude_model_to_openai returns a (model, tier) tuple; the
    # middleware tier drives per-tier effort defaults (BIG/MIDDLE/SMALL_MODEL_EFFORT).
    openai_model, model_tier = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Collect system prompt text. Claude accepts system instructions via the
    # top-level `system` field and also as standalone `{"role": "system"}`
    # messages. We gather both here and fold them into the request below.
    system_parts: List[str] = []

    if claude_request.system:
        if isinstance(claude_request.system, str):
            system_parts.append(claude_request.system)
        elif isinstance(claude_request.system, list):
            for block in claude_request.system:
                if isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                    system_parts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    system_parts.append(block.text)

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            openai_message = convert_claude_assistant_message(msg)
            openai_messages.append(openai_message)

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        block.type == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                        if hasattr(block, "type")
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)
        elif msg.role == Constants.ROLE_SYSTEM:
            # Fold message-level system role into the system prompt collected
            # below rather than dropping it (some clients pass system as a
            # message instead of the top-level `system` field).
            system_parts.append(_extract_text(msg.content))

        i += 1

    # Apply the system prompt. Many downstream routers (e.g. NVIDIA NIM Qwen)
    # reject the 'system' role, so prepend the system text to the first user
    # message; if none exists, add a user message carrying it.
    system_text = "\n\n".join(p for p in system_parts if p and p.strip())
    if system_text.strip():
        system_text = system_text.strip()
        merged = False
        for m in openai_messages:
            if m.get("role") == Constants.ROLE_USER:
                existing = m.get("content", "")
                if isinstance(existing, str):
                    m["content"] = system_text + "\n\n" + existing
                    merged = True
                    break
        if not merged:
            openai_messages.insert(0, {"role": Constants.ROLE_USER, "content": system_text})

    # Build OpenAI request
    openai_request = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "stream": claude_request.stream,
    }

    # Trim conversation history to fit within the model's context window.
    # This runs before any parameter enrichment so the trimmed list is final.
    trimmed_messages, clamped_max_tokens = trim_messages_to_fit(
        openai_messages, openai_model, openai_request["max_tokens"]
    )
    openai_request["messages"] = trimmed_messages
    openai_request["max_tokens"] = clamped_max_tokens

    # Reasoning effort routing. normalize_effort resolves the canonical level
    # (the client's /effort value wins; env defaults fall back in the order
    # per-tier BIG/MIDDLE/SMALL_MODEL_EFFORT > global REASONING_EFFORT >
    # built-in "high"), and resolve_effort renders the provider-specific
    # envelope: a top-level
    # reasoning_effort (string or numeric) for NIM GLM / gpt-oss / o-series /
    # deepseek, chat_template_kwargs for Nemotron, or nothing for always-on
    # thinking and non-reasoning models.
    canonical = normalize_effort(
        claude_request.reasoning_effort, config.get_effort_default(model_tier)
    )
    if _is_reasoning_model(openai_model):
        spec = resolve_effort(openai_model, canonical)
        if spec.reasoning_effort is not None:
            openai_request["reasoning_effort"] = spec.reasoning_effort
        if spec.chat_template_kwargs:
            # Threaded through the OpenAI SDK's extra_body merge, which folds
            # these keys into the outgoing request body.
            openai_request.setdefault("extra_body", {}).update(
                {"chat_template_kwargs": spec.chat_template_kwargs}
            )

    # A subset of reasoning models (kimi-k3, deepseek-r1, OpenAI o-series)
    # reject temperature/top_p entirely; for those we drop sampling params.
    # Others (deepseek-v4.1-flash, gpt-oss, glm-5) accept both sampling and
    # reasoning controls, so temperature is still sent.
    if not _rejects_sampling_params(openai_model):
        openai_request["temperature"] = claude_request.temperature
    # Log only non-sensitive routing metadata; never log message content
    # (system/user text or base64 image payloads) at any level. The
    # char-counting work is gated on DEBUG level so a debug-only metric
    # can never break request handling and is skipped in production.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Converted Claude -> OpenAI: model=%s, messages=%d, content_chars=%d, "
            "max_tokens=%d, stream=%s",
            openai_model,
            len(openai_messages),
            _message_content_char_len(openai_messages),
            openai_request["max_tokens"],
            openai_request["stream"],
        )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None and not _rejects_sampling_params(openai_model):
        openai_request["top_p"] = claude_request.top_p

    # Convert tools
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            # Claude "any" forces the model to call at least one tool; the
            # OpenAI equivalent is "required", not "auto" (which allows a
            # plain-text reply with no tool call).
            openai_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request


def _is_reasoning_model(model_name: str) -> bool:
    """Return True if the upstream model has a controllable reasoning mode.

    This drives effort routing (resolve_effort). It is a superset of the models
    that reject temperature/top_p — see _rejects_sampling_params. Detecting by
    name is a heuristic: it covers the reasoning-model identifiers this proxy
    targets without needing a per-provider registry.
    """
    name = (model_name or "").lower()
    markers = (
        "kimi-k3",
        "reasoner",
        "reasoning",
        "thinking",
        "-o1",
        "-o3",
        "-o4",
        "o1-",
        "o3-",
        "o4-",
        "o1/",
        "o3/",
        "o4/",
        "deepseek-r1",
        "deepseek/r1",
        "deepseek-v3.2",
        "deepseek-v4",
        "-r1",
        "_r1",
        "/r1",
        "r1-",
        "r1_",
        # Reasoning families this proxy routes effort to (see effort_mapper).
        "glm-5",
        "z-ai",
        "gpt-oss",
        "nemotron",
    )
    return any(m in name for m in markers)


def _rejects_sampling_params(model_name: str) -> bool:
    """Return True for reasoning models that reject temperature/top_p.

    Only some reasoning endpoints hard-reject sampling params (kimi-k3,
    deepseek-r1, OpenAI o-series). Others (deepseek-v4.1-flash, gpt-oss, glm-5,
    Nemotron) accept temperature alongside their reasoning control, so those
    must NOT match here.
    """
    name = (model_name or "").lower()
    markers = (
        "kimi-k3",
        "reasoner",
        "reasoning",
        "-o1",
        "-o3",
        "-o4",
        "o1-",
        "o3-",
        "o4-",
        "o1/",
        "o3/",
        "o4/",
        "deepseek-r1",
        "deepseek/r1",
        "-r1",
        "_r1",
        "/r1",
        "r1-",
        "r1_",
    )
    return any(m in name for m in markers)


def _message_content_char_len(messages: List[Dict[str, Any]]) -> int:
    """Return the total character count of text in OpenAI-format messages.

    Used only as a non-sensitive size hint for DEBUG logging: it counts text
    length across every message while ignoring base64 image payloads and tool
    call arguments. Handles all content shapes produced by the converter
    (plain string, ``None``, and lists of text/``image_url`` part dicts)
    without raising, so a debug-only metric can never break request handling.
    """
    total = 0
    for m in messages:
        content = m.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == Constants.CONTENT_TEXT
                    and isinstance(part.get("text"), str)
                ):
                    total += len(part["text"])
    return total


def _extract_text(content: Any) -> str:
    """Flatten Claude message content into a plain string.

    Accepts the union of content shapes a ``ClaudeMessage`` may hold
    (``None``, ``str``, or a list of content blocks) and returns the joined
    text of any text blocks. Used to turn ``{"role": "system"}`` messages
    into system-prompt text. Returns ``""`` for non-text / unsupported
    content rather than raising.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            openai_content.append({"type": "text", "text": block.text})
        elif block.type == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            if (
                isinstance(block.source, dict)
                and block.source.get("type") == "base64"
                and "media_type" in block.source
                and "data" in block.source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    elif openai_content:
        return {"role": Constants.ROLE_USER, "content": openai_content}
    else:
        # No convertible parts (e.g. only URL-sourced images, which this
        # proxy doesn't support yet). Fall back to an empty string rather
        # than emitting content: [], which most routers reject as a
        # non-empty validation error.
        return {"role": Constants.ROLE_USER, "content": ""}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format."""
    text_parts = []
    tool_calls = []

    if msg.content is None:
        # Empty string keeps the field present for routers that reject
        # content: null on an assistant message.
        return {"role": Constants.ROLE_ASSISTANT, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_ASSISTANT, "content": msg.content}

    for block in msg.content:
        if block.type == Constants.CONTENT_TEXT:
            text_parts.append(block.text)
        elif block.type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                }
            )

    openai_message = {"role": Constants.ROLE_ASSISTANT}

    # Set content. Use an empty string (never None) when there is no text so
    # routers that reject content: null on an assistant message accept the
    # turn; tool_calls carry any invoked tools separately.
    if text_parts:
        openai_message["content"] = "\n".join(text_parts)
    else:
        openai_message["content"] = ""

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai_message


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            if block.type == Constants.CONTENT_TOOL_RESULT:
                content = parse_tool_result_content(block.content)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
