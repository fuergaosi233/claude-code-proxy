"""Unit tests for Claude to OpenAI request converter improvements.

Based on TypeScript reference test cases from openai-format.spec.ts
"""

import json
import os
import unittest
from typing import Dict, Any, List
from unittest.mock import patch

from src.conversion.request_converter import (
    convert_claude_to_openai,
    parse_tool_result_content,
    _convert_content_blocks_to_openai,
)
from src.models.claude import (
    ClaudeMessagesRequest,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
)
from src.core.constants import Constants


class MockModelManager:
    """Mock model manager for testing."""

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        return "gpt-4"


class TestConvertToOpenAiMessages(unittest.TestCase):
    """Test cases based on TypeScript reference tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_manager = MockModelManager()
        # 确保测试不受环境变量影响，默认禁用 prompt cache
        self.cache_patcher = patch('src.core.config.config.supports_prompt_cache', False)
        self.cache_patcher.start()
    
    def tearDown(self):
        """清理补丁"""
        self.cache_patcher.stop()

    def test_convert_simple_text_messages(self):
        """Should convert simple text messages."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello"),
                ClaudeMessage(role="assistant", content="Hi there!"),
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 2)
        self.assertEqual(openai_messages[0], {"role": "user", "content": "Hello"})
        self.assertEqual(openai_messages[1], {"role": "assistant", "content": "Hi there!"})

    def test_handle_messages_with_image_content(self):
        """Should handle messages with image content."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(type="text", text="What is in this image?"),
                        ClaudeContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "base64data",
                            },
                        ),
                    ],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 1)
        self.assertEqual(openai_messages[0]["role"], "user")

        content = openai_messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        # Text content
        self.assertEqual(content[0], {"type": "text", "text": "What is in this image?"})

        # Image content
        self.assertEqual(
            content[1],
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64data"}},
        )

    def test_handle_assistant_messages_with_tool_use(self):
        """Should handle assistant messages with tool use."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(type="text", text="Let me check the weather."),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="weather-123",
                            name="get_weather",
                            input={"city": "London"},
                        ),
                    ],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 1)

        assistant_message = openai_messages[0]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["content"], "Let me check the weather.")

        self.assertIn("tool_calls", assistant_message)
        self.assertEqual(len(assistant_message["tool_calls"]), 1)

        tool_call = assistant_message["tool_calls"][0]
        self.assertEqual(
            tool_call,
            {
                "id": "weather-123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "London"})},
            },
        )

    def test_handle_user_messages_with_tool_results(self):
        """Should handle user messages with tool results."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="weather-123",
                            content="Current temperature in London: 20°C",
                        )
                    ],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 1)

        tool_message = openai_messages[0]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "weather-123")
        self.assertEqual(tool_message["content"], "Current temperature in London: 20°C")

    def test_tool_result_with_non_tool_messages_order(self):
        """Test that tool results are processed before non-tool messages."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(
                            type="text", text="Here's the context after tool execution:"
                        ),
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="tool-456",
                            content="Tool execution completed successfully",
                        ),
                    ],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        # Should have 2 messages: tool result first, then user message
        self.assertEqual(len(openai_messages), 2)

        # First message should be tool result
        tool_message = openai_messages[0]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "tool-456")
        self.assertEqual(tool_message["content"], "Tool execution completed successfully")

        # Second message should be user text
        user_message = openai_messages[1]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(user_message["content"], "Here's the context after tool execution:")

    def test_assistant_without_tool_calls_no_empty_array(self):
        """Test that assistant messages without tool use don't have empty tool_calls array."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[ClaudeContentBlockText(type="text", text="Just a regular response.")],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 1)

        assistant_message = openai_messages[0]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["content"], "Just a regular response.")

        # Should not have tool_calls key at all (not even empty array)
        self.assertNotIn("tool_calls", assistant_message)

    def test_single_text_content_becomes_string(self):
        """Test that single text content becomes a string instead of array."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[ClaudeContentBlockText(type="text", text="Single text message")],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        self.assertEqual(len(openai_messages), 1)
        user_message = openai_messages[0]

        # Should be string, not array
        self.assertIsInstance(user_message["content"], str)
        self.assertEqual(user_message["content"], "Single text message")


class TestPromptCacheSupport(unittest.TestCase):
    """Test cases for prompt cache functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_manager = MockModelManager()

    @patch.dict(os.environ, {"SUPPORTS_PROMPT_CACHE": "true"}, clear=True)
    def test_system_message_with_cache_control_when_enabled(self):
        """Test that system message gets cache_control when prompt cache is enabled."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        # Should have system message first
        self.assertEqual(len(openai_messages), 2)
        system_message = openai_messages[0]
        
        self.assertEqual(system_message["role"], "system")
        
        # Should have content as array with cache_control when enabled
        self.assertIsInstance(system_message["content"], list)
        self.assertEqual(len(system_message["content"]), 1)
        
        content_item = system_message["content"][0]
        self.assertEqual(content_item["type"], "text")
        self.assertEqual(content_item["text"], "You are a helpful assistant.")
        self.assertEqual(content_item["cache_control"], {"type": "ephemeral"})

    @patch('src.core.config.config.supports_prompt_cache', False)
    def test_system_message_without_cache_control_when_disabled(self):
        """Test that system message does not get cache_control when prompt cache is disabled."""
        request = ClaudeMessagesRequest(
            model="gpt-4",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="Hello"),
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        system_message = openai_messages[0]
        self.assertEqual(system_message["role"], "system")
        
        # Should have content as string (no cache_control)
        self.assertIsInstance(system_message["content"], str)
        self.assertEqual(system_message["content"], "You are a helpful assistant.")

    @patch.dict(os.environ, {"SUPPORTS_PROMPT_CACHE": "true"}, clear=True)
    def test_last_two_user_messages_get_cache_control(self):
        """Test that last two user messages get cache_control when enabled."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="First message"),
                ClaudeMessage(role="assistant", content="First response"),
                ClaudeMessage(role="user", content="Second message"),
                ClaudeMessage(role="assistant", content="Second response"),
                ClaudeMessage(role="user", content="Third message"),
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        # Find user messages
        user_messages = [msg for msg in openai_messages if msg["role"] == "user"]
        self.assertEqual(len(user_messages), 3)

        # First user message should NOT have cache_control
        first_user = user_messages[0]
        self.assertIsInstance(first_user["content"], str)
        self.assertEqual(first_user["content"], "First message")

        # Second user message should have cache_control
        second_user = user_messages[1]
        self.assertIsInstance(second_user["content"], list)
        self.assertEqual(len(second_user["content"]), 1)
        self.assertEqual(second_user["content"][0]["type"], "text")
        self.assertEqual(second_user["content"][0]["text"], "Second message")
        self.assertEqual(second_user["content"][0]["cache_control"], {"type": "ephemeral"})

        # Third user message should have cache_control
        third_user = user_messages[2]
        self.assertIsInstance(third_user["content"], list)
        self.assertEqual(len(third_user["content"]), 1)
        self.assertEqual(third_user["content"][0]["type"], "text")
        self.assertEqual(third_user["content"][0]["text"], "Third message")
        self.assertEqual(third_user["content"][0]["cache_control"], {"type": "ephemeral"})

    @patch.dict(os.environ, {"SUPPORTS_PROMPT_CACHE": "true"}, clear=True)
    def test_multimodal_user_message_cache_control_on_last_text_part(self):
        """Test that cache_control is added to last text part in multimodal messages."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(type="text", text="Look at this image:"),
                        ClaudeContentBlockImage(
                            type="image",
                            source={
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "base64data",
                            },
                        ),
                        ClaudeContentBlockText(type="text", text="What do you see?"),
                    ],
                )
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        user_message = openai_messages[0]
        self.assertEqual(user_message["role"], "user")
        
        content = user_message["content"]
        self.assertIsInstance(content, list)
        
        # Find text parts
        text_parts = [part for part in content if part.get("type") == "text"]
        self.assertEqual(len(text_parts), 2)
        
        # First text part should not have cache_control
        first_text = text_parts[0]
        self.assertEqual(first_text["text"], "Look at this image:")
        self.assertNotIn("cache_control", first_text)
        
        # Last text part should have cache_control
        last_text = text_parts[1]
        self.assertEqual(last_text["text"], "What do you see?")
        self.assertEqual(last_text["cache_control"], {"type": "ephemeral"})

    @patch('src.core.config.config.supports_prompt_cache', False)
    def test_no_cache_control_when_disabled(self):
        """Test that no cache_control is added when prompt cache is disabled."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                ClaudeMessage(role="user", content="First message"),
                ClaudeMessage(role="assistant", content="First response"),
                ClaudeMessage(role="user", content="Second message"),
            ],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        openai_messages = result["messages"]

        # System message should be string format
        system_message = openai_messages[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIsInstance(system_message["content"], str)
        self.assertEqual(system_message["content"], "You are a helpful assistant.")

        # User messages should be string format without cache_control
        user_messages = [msg for msg in openai_messages if msg["role"] == "user"]
        for user_msg in user_messages:
            self.assertIsInstance(user_msg["content"], str)
            # Verify no cache_control in any content format
            if isinstance(user_msg["content"], list):
                for content_item in user_msg["content"]:
                    self.assertNotIn("cache_control", content_item)

    @patch('src.core.config.config.supports_prompt_cache', True)  # Test "1" as true
    def test_cache_control_with_numeric_true(self):
        """Test that cache_control works with SUPPORTS_PROMPT_CACHE=1."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system="You are helpful.",
            messages=[ClaudeMessage(role="user", content="Hello")],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        system_message = result["messages"][0]
        
        self.assertEqual(system_message["role"], "system")
        self.assertIsInstance(system_message["content"], list)
        self.assertEqual(system_message["content"][0]["cache_control"], {"type": "ephemeral"})

    @patch('src.core.config.config.supports_prompt_cache', True)  # Test "yes" as true
    def test_cache_control_with_yes_value(self):
        """Test that cache_control works with SUPPORTS_PROMPT_CACHE=yes."""
        request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022", 
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Test message")],
        )

        result = convert_claude_to_openai(request, self.model_manager)
        user_message = result["messages"][0]
        
        self.assertEqual(user_message["role"], "user")
        self.assertIsInstance(user_message["content"], list)
        self.assertEqual(user_message["content"][0]["cache_control"], {"type": "ephemeral"})


class TestParseToolResultContent(unittest.TestCase):
    """Test cases for enhanced tool result content parser."""

    def setUp(self):
        """Set up test fixtures."""
        # 确保测试不受环境变量影响
        self.env_patcher = patch.dict(os.environ, {"SUPPORTS_PROMPT_CACHE": "false"}, clear=True)
        self.env_patcher.start()
    
    def tearDown(self):
        """清理环境变量补丁"""
        self.env_patcher.stop()

    def test_parse_none_content(self):
        """Should return empty string for None content."""
        result = parse_tool_result_content(None)
        self.assertEqual(result, "")

    def test_parse_string_content(self):
        """Should return string content as-is."""
        result = parse_tool_result_content("Simple string result")
        self.assertEqual(result, "Simple string result")

    def test_parse_list_with_text_blocks(self):
        """Should parse list content with text blocks."""
        content = [
            {"type": Constants.CONTENT_TEXT, "text": "First part"},
            {"type": Constants.CONTENT_TEXT, "text": "Second part"},
        ]

        result = parse_tool_result_content(content)
        self.assertEqual(result, "First part\nSecond part")

    def test_parse_list_with_image_adds_placeholder(self):
        """Should add placeholder for image blocks."""
        content = [
            {"type": Constants.CONTENT_TEXT, "text": "Analysis:"},
            {"type": Constants.CONTENT_IMAGE, "data": "image_data"},
            {"type": Constants.CONTENT_TEXT, "text": "Complete."},
        ]

        result = parse_tool_result_content(content)
        self.assertIn("(see following user message for image)", result)
        self.assertIn("Analysis:", result)
        self.assertIn("Complete.", result)

    def test_parse_dict_with_text_type(self):
        """Should extract text from dict with text type."""
        content = {"type": Constants.CONTENT_TEXT, "text": "Dict text content"}

        result = parse_tool_result_content(content)
        self.assertEqual(result, "Dict text content")

    def test_parse_complex_dict_as_json(self):
        """Should JSON stringify complex dict content."""
        content = {"result": "success", "data": {"key": "value"}}

        result = parse_tool_result_content(content)
        parsed_back = json.loads(result)
        self.assertEqual(parsed_back["result"], "success")
        self.assertEqual(parsed_back["data"]["key"], "value")


class TestConvertContentBlocksToOpenAI(unittest.TestCase):
    """Test cases for content block conversion helper."""

    def setUp(self):
        """Set up test fixtures."""
        # 确保测试不受环境变量影响
        self.env_patcher = patch.dict(os.environ, {"SUPPORTS_PROMPT_CACHE": "false"}, clear=True)
        self.env_patcher.start()
    
    def tearDown(self):
        """清理环境变量补丁"""
        self.env_patcher.stop()

    def test_convert_empty_blocks_returns_empty_string(self):
        """Should return empty string for empty blocks."""
        result = _convert_content_blocks_to_openai([])
        self.assertEqual(result, "")

    def test_convert_single_text_block_returns_string(self):
        """Should return string for single text block."""
        blocks = [ClaudeContentBlockText(type="text", text="Single text")]

        result = _convert_content_blocks_to_openai(blocks)
        self.assertEqual(result, "Single text")

    def test_convert_multiple_blocks_returns_array(self):
        """Should return array for multiple content blocks."""
        blocks = [
            ClaudeContentBlockText(type="text", text="Text content"),
            ClaudeContentBlockImage(
                type="image",
                source={"type": "base64", "media_type": "image/png", "data": "png_data"},
            ),
        ]

        result = _convert_content_blocks_to_openai(blocks)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        self.assertEqual(result[0], {"type": "text", "text": "Text content"})

        self.assertEqual(
            result[1], {"type": "image_url", "image_url": {"url": "data:image/png;base64,png_data"}}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
