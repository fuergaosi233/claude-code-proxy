"""Test script for Claude to OpenAI proxy."""

import asyncio
import json

import httpx
from dotenv import load_dotenv

load_dotenv()


async def test_basic_chat():
    """Test basic chat completion."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )

        print("Basic chat response:")
        print(json.dumps(response.json(), indent=2))


async def test_streaming_chat():
    """Test streaming chat completion."""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": "Tell me a short joke"}],
                "stream": True,
            },
        ) as response:
            print("\nStreaming response:")
            async for line in response.aiter_lines():
                if line.strip():
                    print(line)


async def test_function_calling():
    """Test function calling capability."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather like in New York? Please use the weather function.",
                    }
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get weather for",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "Temperature unit",
                                },
                            },
                            "required": ["location"],
                        },
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
        )

        print("\nFunction calling response:")
        print(json.dumps(response.json(), indent=2))


async def test_with_system_message():
    """Test with system message."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "system": "You are a helpful assistant that always responds in haiku format.",
                "messages": [{"role": "user", "content": "Explain what AI is"}],
            },
        )

        print("\nSystem message response:")
        print(json.dumps(response.json(), indent=2))


async def test_multimodal():
    """Test multimodal input (text + image)."""
    async with httpx.AsyncClient() as client:
        # Sample base64 image (1x1 pixel transparent PNG)
        sample_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU8PJAAAAASUVORK5CYII="

        response = await client.post(
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see in this image?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": sample_image,
                                },
                            },
                        ],
                    }
                ],
            },
        )

        print("\nMultimodal response:")
        print(json.dumps(response.json(), indent=2))


async def test_conversation_with_tool_use():
    """Test a complete conversation with tool use and results."""
    async with httpx.AsyncClient() as client:
        # First message with tool call
        response1 = await client.post(
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "Calculate 25 * 4 using the calculator tool"}
                ],
                "tools": [
                    {
                        "name": "calculator",
                        "description": "Perform basic arithmetic calculations",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to calculate",
                                }
                            },
                            "required": ["expression"],
                        },
                    }
                ],
            },
        )

        print("\nTool call response:")
        result1 = response1.json()
        print(json.dumps(result1, indent=2))

        # Simulate tool execution and send result
        if result1.get("content"):
            tool_use_blocks = [
                block for block in result1["content"] if block.get("type") == "tool_use"
            ]
            if tool_use_blocks:
                tool_block = tool_use_blocks[0]

                # Second message with tool result
                response2 = await client.post(
                    "http://localhost:8082/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 100,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Calculate 25 * 4 using the calculator tool",
                            },
                            {"role": "assistant", "content": result1["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_block["id"],
                                        "content": "100",
                                    }
                                ],
                            },
                        ],
                    },
                )

                print("\nTool result response:")
                print(json.dumps(response2.json(), indent=2))


async def test_token_counting():
    """Test token counting endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8082/v1/messages/count_tokens",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "user", "content": "This is a test message for token counting."}
                ],
            },
        )

        print("\nToken count response:")
        print(json.dumps(response.json(), indent=2))


async def test_health_and_connection():
    """Test health and connection endpoints."""
    async with httpx.AsyncClient() as client:
        # Health check
        health_response = await client.get("http://localhost:8082/health")
        print("\nHealth check:")
        print(json.dumps(health_response.json(), indent=2))

        # Connection test
        connection_response = await client.get("http://localhost:8082/test-connection")
        print("\nConnection test:")
        print(json.dumps(connection_response.json(), indent=2))


async def main():
    """Run all tests."""
    print("🧪 Testing Claude to OpenAI Proxy")
    print("=" * 50)

    try:
        await test_health_and_connection()
        await test_token_counting()
        await test_basic_chat()
        await test_with_system_message()
        await test_streaming_chat()
        await test_multimodal()
        await test_function_calling()
        await test_conversation_with_tool_use()

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the server is running with a valid OPENAI_API_KEY")


if __name__ == "__main__":
    asyncio.run(main())
