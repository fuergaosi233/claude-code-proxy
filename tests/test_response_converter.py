"""Unit tests for OpenAI to Claude response converter improvements.

Test the handling of image content in OpenAI responses to prevent context window overflow.
"""

import json
import unittest
from typing import Dict, Any

from src.conversion.response_converter import convert_openai_to_claude_response
from src.models.claude import ClaudeMessagesRequest
from src.core.constants import Constants


class TestConvertOpenAIToClaude(unittest.TestCase):
    """Test cases for OpenAI to Claude response conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[]
        )

    def test_convert_simple_text_response(self):
        """Should convert simple text response correctly."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        self.assertEqual(result["role"], Constants.ROLE_ASSISTANT)
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][0]["text"], "Hello, how can I help you?")

    def test_convert_structured_content_with_text_and_image(self):
        """Should convert structured content with text and image correctly."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the image you requested:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
                                }
                            }
                        ],
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 30
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Should have 2 content blocks
        self.assertEqual(len(result["content"]), 2)
        
        # First block should be text
        self.assertEqual(result["content"][0]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][0]["text"], "Here is the image you requested:")
        
        # Second block should be image
        self.assertEqual(result["content"][1]["type"], Constants.CONTENT_IMAGE)
        self.assertEqual(result["content"][1]["source"]["type"], "base64")
        self.assertEqual(result["content"][1]["source"]["media_type"], "image/jpeg")
        self.assertEqual(result["content"][1]["source"]["data"], "/9j/4AAQSkZJRgABAQAAAQABAAD")


class TestHandleImageContent(unittest.TestCase):
    """Test cases for image content handling in responses."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[]
        )

    def test_convert_structured_content_with_malformed_data_url(self):
        """Should handle malformed data URLs gracefully."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Some text"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:malformed-without-comma"
                                }
                            }
                        ],
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Should have 2 content blocks
        self.assertEqual(len(result["content"]), 2)
        
        # First block should be text
        self.assertEqual(result["content"][0]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][0]["text"], "Some text")
        
        # Second block should be a text placeholder for malformed data URL
        self.assertEqual(result["content"][1]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][1]["text"], "[Image content]")

    def test_convert_structured_content_with_external_url(self):
        """Should handle external image URLs as text placeholders."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": "Some text"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image.jpg"
                                }
                            }
                        ],
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Should have 2 content blocks
        self.assertEqual(len(result["content"]), 2)
        
        # First block should be text
        self.assertEqual(result["content"][0]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][0]["text"], "Some text")
        
        # Second block should be a text placeholder for external image URL
        self.assertEqual(result["content"][1]["type"], Constants.CONTENT_TEXT)
        self.assertEqual(result["content"][1]["text"], "[Image: https://example.com/image.jpg]")


class TestCacheTokenHandling(unittest.TestCase):
    """Test cases for handling OpenAI cache token data."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[]
        )
        # Mock config.supports_prompt_cache to return True for these tests
        import unittest.mock
        self.cache_patcher = unittest.mock.patch('src.core.config.config.supports_prompt_cache', True)
        self.cache_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.cache_patcher.stop()

    def test_convert_response_with_openai_cached_tokens(self):
        """Should correctly map OpenAI cached_tokens to Claude cache_read_input_tokens."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 328,
                "input_tokens_details": {
                    "cached_tokens": 250
                },
                "output_tokens": 52,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 380
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that cache tokens are correctly mapped
        self.assertEqual(result["usage"]["input_tokens"], 328)
        self.assertEqual(result["usage"]["output_tokens"], 52)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 250)
        self.assertNotIn("cache_creation_input_tokens", result["usage"])

    def test_convert_response_with_no_openai_cache_data_large_request(self):
        """Should not include cache tokens when no cache data provided by OpenAI."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that basic token counts are preserved
        self.assertEqual(result["usage"]["input_tokens"], 100)
        self.assertEqual(result["usage"]["output_tokens"], 20)
        # No cache tokens should be present when OpenAI doesn't provide cache data
        self.assertNotIn("cache_creation_input_tokens", result["usage"])
        self.assertNotIn("cache_read_input_tokens", result["usage"])

    def test_convert_response_with_no_openai_cache_data_small_request(self):
        """Should not include cache tokens when no cache data provided by OpenAI."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hi",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that basic token counts are preserved
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 2)
        # No cache tokens should be present when OpenAI doesn't provide cache data
        self.assertNotIn("cache_creation_input_tokens", result["usage"])
        self.assertNotIn("cache_read_input_tokens", result["usage"])

    def test_convert_response_with_zero_cached_tokens(self):
        """Should not include cache tokens when OpenAI reports zero cached tokens."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 75,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 25,
                "total_tokens": 100
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Should not include cache tokens when cached_tokens is 0
        self.assertEqual(result["usage"]["input_tokens"], 75)
        self.assertEqual(result["usage"]["output_tokens"], 25)
        self.assertNotIn("cache_creation_input_tokens", result["usage"])
        self.assertNotIn("cache_read_input_tokens", result["usage"])

    def test_convert_response_with_direct_openai_cache_tokens(self):
        """Should correctly map OpenAI direct cache_creation_input_tokens and cache_read_input_tokens."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 328,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
                "output_tokens": 52,
                "total_tokens": 380
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that direct cache tokens are correctly mapped
        self.assertEqual(result["usage"]["input_tokens"], 328)
        self.assertEqual(result["usage"]["output_tokens"], 52)
        self.assertEqual(result["usage"]["cache_creation_input_tokens"], 100)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 200)

    def test_convert_response_with_only_cache_read_tokens(self):
        """Should handle response with only cache_read_input_tokens."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 328,
                "cache_read_input_tokens": 250,
                "output_tokens": 52,
                "total_tokens": 380
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that only cache_read_input_tokens is included
        self.assertEqual(result["usage"]["input_tokens"], 328)
        self.assertEqual(result["usage"]["output_tokens"], 52)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 250)
        self.assertNotIn("cache_creation_input_tokens", result["usage"])

    def test_convert_response_with_direct_and_legacy_cache_tokens(self):
        """Should prefer direct cache tokens over legacy format when both are present."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "input_tokens": 328,
                "cache_read_input_tokens": 200,  # Direct field should take priority
                "input_tokens_details": {
                    "cached_tokens": 150  # Legacy field should be ignored
                },
                "output_tokens": 52,
                "total_tokens": 380
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Check that direct cache tokens take priority over legacy
        self.assertEqual(result["usage"]["input_tokens"], 328)
        self.assertEqual(result["usage"]["output_tokens"], 52)
        self.assertEqual(result["usage"]["cache_read_input_tokens"], 200)  # Should use direct field
        self.assertNotIn("cache_creation_input_tokens", result["usage"])


class TestHandleUnknownContent(unittest.TestCase):
    """Test cases for handling unknown content types."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[]
        )

    def test_convert_unknown_content_type(self):
        """Should handle unknown content types gracefully."""
        openai_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "unknown_type",
                                "data": "some data"
                            }
                        ],
                        "role": Constants.ROLE_ASSISTANT
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }

        result = convert_openai_to_claude_response(openai_response, self.original_request)

        # Should convert unknown type to text
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], Constants.CONTENT_TEXT)
        self.assertIn("unknown_type", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()