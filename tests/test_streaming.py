import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.conversion.response_converter import (
    _convert_openai_streaming_core,
    convert_openai_streaming_to_claude,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest


def _make_request(model="claude-3-5-sonnet-20241022"):
    return ClaudeMessagesRequest(
        model=model,
        max_tokens=1024,
        messages=[ClaudeMessage(role="user", content="Hello")],
        stream=True,
    )


async def _make_stream(lines):
    """Create an async generator that yields lines."""
    for line in lines:
        yield line


async def _collect_events(async_gen):
    """Collect all events from an async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


class TestStreamingBasicText:
    """Tests for basic text streaming conversion."""

    @pytest.mark.asyncio
    async def test_basic_text_streaming(self):
        """Test conversion of a simple text streaming response."""
        logger = MagicMock()
        request = _make_request()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude(_make_stream(stream_lines), request, logger)
        )

        # Check initial events
        assert "message_start" in events[0]
        assert "content_block_start" in events[1]
        assert "ping" in events[2]

        # Check text deltas
        text_deltas = [e for e in events if "text_delta" in e]
        assert len(text_deltas) == 2

        # Check first text delta has "Hello"
        delta_data = json.loads(text_deltas[0].split("data: ")[1].rstrip("\n"))
        assert delta_data["delta"]["text"] == "Hello"

        # Check final events
        assert "content_block_stop" in events[-3]
        assert "message_delta" in events[-2]
        assert "message_stop" in events[-1]

    @pytest.mark.asyncio
    async def test_sse_event_sequence(self):
        """Verify the complete SSE event sequence."""
        logger = MagicMock()
        request = _make_request()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude(_make_stream(stream_lines), request, logger)
        )

        event_types = []
        for e in events:
            if "event: " in e:
                event_type = e.split("event: ")[1].split("\n")[0]
                event_types.append(event_type)

        assert event_types[0] == "message_start"
        assert event_types[1] == "content_block_start"
        assert event_types[2] == "ping"
        assert "content_block_delta" in event_types
        assert event_types[-3] == "content_block_stop"
        assert event_types[-2] == "message_delta"
        assert event_types[-1] == "message_stop"


class TestStreamingToolCalls:
    """Tests for tool call streaming conversion."""

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        """Test conversion of a tool call streaming response."""
        logger = MagicMock()
        request = _make_request()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\": \\"NYC\\"}"}}]},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude(_make_stream(stream_lines), request, logger)
        )

        # Check that tool content block start was emitted
        tool_starts = [e for e in events if "tool_use" in e and "content_block_start" in e]
        assert len(tool_starts) == 1

        # Check tool input JSON delta was emitted
        json_deltas = [e for e in events if "input_json_delta" in e]
        assert len(json_deltas) == 1

        # Check message_delta has tool_use stop reason
        msg_delta = [e for e in events if "message_delta" in e][0]
        msg_data = json.loads(msg_delta.split("data: ")[1].rstrip("\n"))
        assert msg_data["delta"]["stop_reason"] == "tool_use"


class TestStreamingWithCancellation:
    """Tests for streaming with cancellation support."""

    @pytest.mark.asyncio
    async def test_cancellation_on_disconnect(self):
        """Test that client disconnection triggers cancellation."""
        logger = MagicMock()
        request = _make_request()

        disconnect_called = False

        async def mock_is_disconnected():
            nonlocal disconnect_called
            if not disconnect_called:
                disconnect_called = True
                return False
            return True  # Disconnect on second call

        http_request = MagicMock()
        http_request.is_disconnected = mock_is_disconnected

        openai_client = MagicMock()
        openai_client.cancel_request = MagicMock()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _make_stream(stream_lines),
                request,
                logger,
                http_request,
                openai_client,
                "request-123",
            )
        )

        # Should have called cancel_request
        openai_client.cancel_request.assert_called_once_with("request-123")

    @pytest.mark.asyncio
    async def test_no_cancellation_when_connected(self):
        """Test normal flow when client stays connected."""
        logger = MagicMock()
        request = _make_request()

        http_request = MagicMock()
        http_request.is_disconnected = AsyncMock(return_value=False)

        openai_client = MagicMock()
        openai_client.cancel_request = MagicMock()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _make_stream(stream_lines),
                request,
                logger,
                http_request,
                openai_client,
                "request-456",
            )
        )

        # cancel should NOT be called
        openai_client.cancel_request.assert_not_called()
        # Should have normal event sequence
        assert any("message_stop" in e for e in events)


class TestStreamingUsageExtraction:
    """Tests for usage data extraction in streaming."""

    @pytest.mark.asyncio
    async def test_usage_data_extracted(self):
        """Test that usage data is extracted from streaming chunks."""
        logger = MagicMock()
        request = _make_request()

        stream_lines = [
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":3}}}',
            "data: [DONE]",
        ]

        http_request = MagicMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()
        openai_client.cancel_request = MagicMock()

        events = await _collect_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _make_stream(stream_lines),
                request,
                logger,
                http_request,
                openai_client,
                "req-1",
            )
        )

        # Find the message_delta event
        msg_delta = [e for e in events if "message_delta" in e][0]
        msg_data = json.loads(msg_delta.split("data: ")[1].rstrip("\n"))
        usage = msg_data["usage"]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["cache_read_input_tokens"] == 3


class TestStreamingErrorHandling:
    """Tests for streaming error handling."""

    @pytest.mark.asyncio
    async def test_streaming_error_emits_error_event(self):
        """Test that streaming errors emit an error SSE event."""
        logger = MagicMock()
        request = _make_request()

        async def error_stream():
            yield 'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}'
            raise RuntimeError("Network failure")

        events = await _collect_events(
            convert_openai_streaming_to_claude(error_stream(), request, logger)
        )

        error_events = [e for e in events if "event: error" in e]
        assert len(error_events) == 1
        assert "Network failure" in error_events[0]

    @pytest.mark.asyncio
    async def test_malformed_json_chunk_skipped(self):
        """Test that malformed JSON chunks are skipped with a warning."""
        logger = MagicMock()
        request = _make_request()

        stream_lines = [
            "data: {invalid json",
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}',
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}',
            "data: [DONE]",
        ]

        events = await _collect_events(
            convert_openai_streaming_to_claude(_make_stream(stream_lines), request, logger)
        )

        # Should still emit text delta for valid chunk
        text_deltas = [e for e in events if "text_delta" in e]
        assert len(text_deltas) == 1
        # Logger warning should have been called for malformed chunk
        logger.warning.assert_called()
