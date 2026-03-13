import json

import pytest

from src.conversion.response_converter import convert_openai_to_claude_response
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


class TestConvertOpenaiToClaudeResponse:
    """Tests for the non-streaming response conversion."""

    def _make_request(self, model="claude-3-5-sonnet-20241022", max_tokens=1024):
        return ClaudeMessagesRequest(
            model=model,
            max_tokens=max_tokens,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )

    def test_basic_text_response(self, sample_openai_response):
        request = self._make_request()
        result = convert_openai_to_claude_response(sample_openai_response, request)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello! I'm doing well."
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 8

    def test_tool_call_response(self, sample_openai_tool_response):
        request = self._make_request()
        result = convert_openai_to_claude_response(sample_openai_tool_response, request)

        assert result["stop_reason"] == "tool_use"
        # Should have tool_use content block (no text since content is None)
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"location": "San Francisco"}
        assert tool_blocks[0]["id"] == "call_abc123"

    def test_no_choices_raises_error(self):
        from fastapi import HTTPException

        request = self._make_request()
        response = {"id": "test", "choices": [], "usage": {}}

        with pytest.raises(HTTPException) as exc_info:
            convert_openai_to_claude_response(response, request)
        assert exc_info.value.status_code == 500

    def test_finish_reason_length(self):
        request = self._make_request()
        response = {
            "id": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Partial..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100},
        }
        result = convert_openai_to_claude_response(response, request)
        assert result["stop_reason"] == "max_tokens"

    def test_finish_reason_tool_calls(self):
        request = self._make_request()
        response = {
            "id": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = convert_openai_to_claude_response(response, request)
        assert result["stop_reason"] == "tool_use"

    def test_empty_content_default(self):
        request = self._make_request()
        response = {
            "id": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        result = convert_openai_to_claude_response(response, request)
        # Should have at least one content block (empty text)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_invalid_tool_arguments_json(self):
        request = self._make_request()
        response = {
            "id": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "test",
                                    "arguments": "invalid json{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = convert_openai_to_claude_response(response, request)
        tool_block = result["content"][0]
        assert "raw_arguments" in tool_block["input"]
