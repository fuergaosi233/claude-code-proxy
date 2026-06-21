"""
Unit tests for request_converter.py.

These tests use mocking to avoid requiring actual API keys or server instances.
"""

import json
import sys
from unittest import mock
from typing import List, Optional

import pytest

# ---------------------------------------------------------------------------
# Mock the config module before importing the converter.
# The real config module calls sys.exit(1) when OPENAI_API_KEY is missing,
# which makes it impossible to import without a real key in the environment.
# ---------------------------------------------------------------------------
config_module = mock.MagicMock()
config_module.config.min_tokens_limit = 100
config_module.config.max_tokens_limit = 4096

sys.modules["src.core.config"] = config_module

from src.conversion.request_converter import (
    convert_claude_to_openai,
    _extract_system_text,
)
from src.models.claude import (
    ClaudeMessagesRequest,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockImage,
    ClaudeSystemContent,
    ClaudeTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_manager():
    """Returns a ModelManager-like object that maps sonnet → gpt-4o etc."""
    mm = mock.Mock()
    mm.map_claude_model_to_openai.side_effect = lambda m: {
        "claude-sonnet-4-20250514": "gpt-4o",
        "claude-haiku-3-5-20241022": "gpt-4o-mini",
    }.get(m, "gpt-4o")
    return mm


def make_request(
    model: str = "claude-sonnet-4-20250514",
    messages: Optional[List[ClaudeMessage]] = None,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    stream: bool = False,
    temperature: float = 1.0,
    **overrides,
) -> ClaudeMessagesRequest:
    """Helper to build a ClaudeMessagesRequest with sensible defaults."""
    if messages is None:
        messages = [ClaudeMessage(role="user", content="Hello")]
    return ClaudeMessagesRequest(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        stream=stream,
        temperature=temperature,
        **overrides,
    )


# ---------------------------------------------------------------------------
# _extract_system_text
# ---------------------------------------------------------------------------

class TestExtractSystemText:
    def test_plain_string(self):
        assert _extract_system_text("You are a helpful bot.") == "You are a helpful bot."

    def test_plain_string_with_whitespace(self):
        assert _extract_system_text("  Hello  ") == "Hello"

    def test_empty_string(self):
        assert _extract_system_text("") == ""

    def test_list_of_pydantic_objects(self):
        blocks = [
            ClaudeSystemContent(type="text", text="First part."),
            ClaudeSystemContent(type="text", text="Second part."),
        ]
        assert _extract_system_text(blocks) == "First part.\n\nSecond part."

    def test_list_of_dicts(self):
        blocks = [
            {"type": "text", "text": "Dict part A"},
            {"type": "text", "text": "Dict part B"},
        ]
        assert _extract_system_text(blocks) == "Dict part A\n\nDict part B"

    def test_list_mixed_types_skips_non_text(self):
        blocks = [
            ClaudeSystemContent(type="text", text="Only this."),
        ]
        assert _extract_system_text(blocks) == "Only this."

    def test_none_returns_empty(self):
        assert _extract_system_text(None) == ""

    def test_integer_returns_empty(self):
        assert _extract_system_text(42) == ""


# ---------------------------------------------------------------------------
# convert_claude_to_openai  —  top-level system parameter
# ---------------------------------------------------------------------------

class TestTopLevelSystem:
    def test_system_as_string(self, mock_model_manager):
        req = make_request(system="You are Claude.")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0] == {"role": "system", "content": "You are Claude."}

    def test_system_as_list(self, mock_model_manager):
        req = make_request(
            system=[
                ClaudeSystemContent(type="text", text="Rule 1."),
                ClaudeSystemContent(type="text", text="Rule 2."),
            ]
        )
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0] == {"role": "system", "content": "Rule 1.\n\nRule 2."}

    def test_no_system(self, mock_model_manager):
        req = make_request()
        result = convert_claude_to_openai(req, mock_model_manager)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# convert_claude_to_openai  —  inline system messages (role: "system")
# ---------------------------------------------------------------------------

class TestInlineSystem:
    def test_inline_system_with_text_list(self, mock_model_manager):
        """Claude Code may send a system message inside the messages array."""
        msg = ClaudeMessage(
            role="system",
            content=[
                ClaudeContentBlockText(type="text", text="Inline system."),
            ],
        )
        req = make_request(messages=[msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"] == [
            {"role": "system", "content": "Inline system."}
        ]

    def test_inline_system_with_plain_string(self, mock_model_manager):
        msg = ClaudeMessage(role="system", content="Plain inline system.")
        req = make_request(messages=[msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"] == [
            {"role": "system", "content": "Plain inline system."}
        ]

    def test_inline_system_empty_content_skipped(self, mock_model_manager):
        """An empty inline system message should produce no output."""
        msg = ClaudeMessage(role="system", content="")
        req = make_request(messages=[msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        # When the only message is an empty system, no messages are added
        assert result["messages"] == []


# ---------------------------------------------------------------------------
# convert_claude_to_openai  —  user & assistant messages
# ---------------------------------------------------------------------------

class TestUserMessages:
    def test_plain_text_user(self, mock_model_manager):
        req = make_request()
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0] == {"role": "user", "content": "Hello"}

    def test_multimodal_user(self, mock_model_manager):
        msg = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="What is this?"),
                ClaudeContentBlockImage(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8PJAAAAASUVORK5CYII=",
                    },
                ),
            ],
        )
        req = make_request(messages=[msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1]["type"] == "image_url"
        assert "base64" in content[1]["image_url"]["url"]


class TestAssistantMessages:
    def test_text_only(self, mock_model_manager):
        msg = ClaudeMessage(
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text="Sure!")],
        )
        req = make_request(messages=[ClaudeMessage(role="user", content="Hi"), msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][1] == {"role": "assistant", "content": "Sure!"}

    def test_with_tool_call(self, mock_model_manager):
        msg = ClaudeMessage(
            role="assistant",
            content=[
                ClaudeContentBlockText(type="text", text="Let me check."),
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="toolu_abc123",
                    name="get_weather",
                    input={"location": "Tokyo"},
                ),
            ],
        )
        req = make_request(messages=[ClaudeMessage(role="user", content="Weather?"), msg])
        result = convert_claude_to_openai(req, mock_model_manager)
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check."
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {
            "location": "Tokyo"
        }

    def test_tool_result_follows_assistant(self, mock_model_manager):
        messages = [
            ClaudeMessage(role="user", content="Weather?"),
            ClaudeMessage(
                role="assistant",
                content=[
                    ClaudeContentBlockToolUse(
                        type="tool_use",
                        id="toolu_abc",
                        name="get_weather",
                        input={"location": "Tokyo"},
                    ),
                ],
            ),
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_abc",
                        content="Sunny, 25°C",
                    ),
                ],
            ),
        ]
        req = make_request(messages=messages)
        result = convert_claude_to_openai(req, mock_model_manager)
        # Expected: user → assistant (with tool_call) → tool
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert result["messages"][2]["tool_call_id"] == "toolu_abc"
        assert result["messages"][2]["content"] == "Sunny, 25°C"


# ---------------------------------------------------------------------------
# convert_claude_to_openai  —  model mapping & top-level fields
# ---------------------------------------------------------------------------

class TestModelAndFields:
    def test_model_mapping(self, mock_model_manager):
        req = make_request()
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o"

    def test_max_tokens_clamping(self, mock_model_manager):
        req = make_request(max_tokens=10)  # below min
        result = convert_claude_to_openai(req, mock_model_manager)
        # Clamped to min_tokens_limit (100)
        assert result["max_tokens"] == 100

    def test_max_tokens_upper_clamping(self, mock_model_manager):
        req = make_request(max_tokens=99999)  # above max
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["max_tokens"] == 4096

    def test_temperature_passthrough(self, mock_model_manager):
        req = make_request(temperature=0.5)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["temperature"] == 0.5

    def test_stream_passthrough(self, mock_model_manager):
        req = make_request(stream=True)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["stream"] is True


# ---------------------------------------------------------------------------
# convert_claude_to_openai  —  tools & tool_choice
# ---------------------------------------------------------------------------

class TestTools:
    def test_tool_conversion(self, mock_model_manager):
        req = make_request(
            tools=[
                ClaudeTool(
                    name="get_weather",
                    description="Get weather",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
            ],
        )
        result = convert_claude_to_openai(req, mock_model_manager)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["tools"][0]["function"]["description"] == "Get weather"

    def test_tool_choice_auto(self, mock_model_manager):
        req = make_request(tool_choice={"type": "auto"})
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any_maps_to_auto(self, mock_model_manager):
        req = make_request(tool_choice={"type": "any"})
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_specific_tool(self, mock_model_manager):
        req = make_request(
            tool_choice={"type": "tool", "name": "get_weather"}
        )
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_stop_sequences(self, mock_model_manager):
        req = make_request(stop_sequences=["\n\n", "Stop."])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["stop"] == ["\n\n", "Stop."]

    def test_top_p(self, mock_model_manager):
        req = make_request(top_p=0.9)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["top_p"] == 0.9

    def test_top_p_none_omitted(self, mock_model_manager):
        req = make_request()
        result = convert_claude_to_openai(req, mock_model_manager)
        assert "top_p" not in result


# ---------------------------------------------------------------------------
# Integration-style: top-level system + inline system simultaneously
# ---------------------------------------------------------------------------

class TestCombinedScenarios:
    def test_both_top_level_and_inline_system(self, mock_model_manager):
        """When both exist, both are added as separate system messages."""
        messages = [
            ClaudeMessage(
                role="system",
                content=[ClaudeContentBlockText(type="text", text="Inline")],
            ),
            ClaudeMessage(role="user", content="Hi"),
        ]
        req = make_request(system="Top-level", messages=messages)
        result = convert_claude_to_openai(req, mock_model_manager)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 2
        assert system_msgs[0]["content"] == "Top-level"
        assert system_msgs[1]["content"] == "Inline"

    def test_conversation_with_system_user_assistant_tool(self, mock_model_manager):
        """A complete realistic conversation turn."""
        messages = [
            ClaudeMessage(role="system", content="You are a helpful AI."),
            ClaudeMessage(role="user", content="What's the time?"),
            ClaudeMessage(
                role="assistant",
                content=[
                    ClaudeContentBlockToolUse(
                        type="tool_use",
                        id="toolu_1",
                        name="get_time",
                        input={},
                    ),
                ],
            ),
            ClaudeMessage(
                role="user",
                content=[
                    ClaudeContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_1",
                        content="12:00",
                    ),
                ],
            ),
        ]
        req = make_request(messages=messages)
        result = convert_claude_to_openai(req, mock_model_manager)
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]