import json
from unittest.mock import MagicMock

import pytest

from src.conversion.request_converter import (
    convert_claude_assistant_message,
    convert_claude_to_openai,
    convert_claude_tool_results,
    convert_claude_user_message,
    parse_tool_result_content,
)
from src.models.claude import (
    ClaudeContentBlockImage,
    ClaudeContentBlockText,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeTool,
)


def _make_model_manager(big="gpt-4o", middle="gpt-4o", small="gpt-4o-mini"):
    from src.core.model_manager import ModelManager

    config = MagicMock()
    config.big_model = big
    config.middle_model = middle
    config.small_model = small
    return ModelManager(config)


class TestConvertClaudeToOpenai:
    """Tests for the main convert_claude_to_openai function."""

    def test_basic_text_message(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )
        result = convert_claude_to_openai(request, mm)

        assert result["model"] == "gpt-4o"
        assert result["max_tokens"] == 1024
        assert result["stream"] is False
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_system_message_string(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[ClaudeMessage(role="user", content="Hi")],
        )
        result = convert_claude_to_openai(request, mm)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"

    def test_system_message_list(self, monkeypatch):
        from src.core.config import Config
        from src.models.claude import ClaudeSystemContent

        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=[ClaudeSystemContent(type="text", text="System prompt here")],
            messages=[ClaudeMessage(role="user", content="Hi")],
        )
        result = convert_claude_to_openai(request, mm)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "System prompt here"

    def test_tools_conversion(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="What's the weather?")],
            tools=[
                ClaudeTool(
                    name="get_weather",
                    description="Get weather info",
                    input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
                )
            ],
        )
        result = convert_claude_to_openai(request, mm)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_choice_auto(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hi")],
            tool_choice={"type": "auto"},
        )
        result = convert_claude_to_openai(request, mm)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hi")],
            tool_choice={"type": "any"},
        )
        result = convert_claude_to_openai(request, mm)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_specific_tool(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hi")],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        result = convert_claude_to_openai(request, mm)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "get_weather"

    def test_stop_sequences(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hi")],
            stop_sequences=["END", "STOP"],
        )
        result = convert_claude_to_openai(request, mm)
        assert result["stop"] == ["END", "STOP"]

    def test_top_p(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "4096")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "100")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hi")],
            top_p=0.9,
        )
        result = convert_claude_to_openai(request, mm)
        assert result["top_p"] == 0.9

    def test_max_tokens_clamped_to_limits(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_LIMIT", "2000")
        monkeypatch.setenv("MIN_TOKENS_LIMIT", "500")
        from src.core.config import Config

        cfg = Config()
        monkeypatch.setattr("src.conversion.request_converter.config", cfg)

        mm = _make_model_manager()

        # Test clamping to max
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
            messages=[ClaudeMessage(role="user", content="Hi")],
        )
        result = convert_claude_to_openai(request, mm)
        assert result["max_tokens"] == 2000

        # Test clamping to min
        request2 = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[ClaudeMessage(role="user", content="Hi")],
        )
        result2 = convert_claude_to_openai(request2, mm)
        assert result2["max_tokens"] == 500


class TestConvertClaudeUserMessage:
    def test_string_content(self):
        msg = ClaudeMessage(role="user", content="Hello")
        result = convert_claude_user_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_text_block_content(self):
        msg = ClaudeMessage(
            role="user",
            content=[ClaudeContentBlockText(type="text", text="Hello world")],
        )
        result = convert_claude_user_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "Hello world"

    def test_multimodal_content(self):
        msg = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="What's in this image?"),
                ClaudeContentBlockImage(
                    type="image",
                    source={"type": "base64", "media_type": "image/png", "data": "abc123"},
                ),
            ],
        )
        result = convert_claude_user_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image_url"
        assert "data:image/png;base64,abc123" in result["content"][1]["image_url"]["url"]


class TestConvertClaudeAssistantMessage:
    def test_text_content(self):
        msg = ClaudeMessage(
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text="Hello!")],
        )
        result = convert_claude_assistant_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello!"

    def test_string_content(self):
        msg = ClaudeMessage(role="assistant", content="Just a string")
        result = convert_claude_assistant_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Just a string"

    def test_tool_use_content(self):
        msg = ClaudeMessage(
            role="assistant",
            content=[
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="tool_123",
                    name="get_weather",
                    input={"location": "NYC"},
                )
            ],
        )
        result = convert_claude_assistant_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "tool_123"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_none_content(self):
        # ClaudeMessage model doesn't accept None content directly,
        # but the function handles it when called with a dict-like object
        from unittest.mock import MagicMock

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        result = convert_claude_assistant_message(msg)
        assert result["content"] is None


class TestConvertClaudeToolResults:
    def test_tool_result_message(self):
        msg = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="tool_123",
                    content="The weather is sunny.",
                )
            ],
        )
        result = convert_claude_tool_results(msg)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tool_123"
        assert result[0]["content"] == "The weather is sunny."


class TestParseToolResultContent:
    def test_none_content(self):
        assert parse_tool_result_content(None) == "No content provided"

    def test_string_content(self):
        assert parse_tool_result_content("result") == "result"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        result = parse_tool_result_content(content)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_dict_content_with_text_type(self):
        content = {"type": "text", "text": "Hello"}
        assert parse_tool_result_content(content) == "Hello"

    def test_dict_content_json_fallback(self):
        content = {"key": "value"}
        result = parse_tool_result_content(content)
        assert '"key"' in result
        assert '"value"' in result
