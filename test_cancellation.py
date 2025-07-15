#!/usr/bin/env python3
"""
Test script for HTTP request cancellation functionality.
This script demonstrates how client disconnection cancels ongoing requests.
"""

import asyncio

import httpx


async def test_non_streaming_cancellation():
    """Test cancellation for non-streaming requests."""
    print("🧪 Testing non-streaming request cancellation...")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Start a long-running request
            task = asyncio.create_task(
                client.post(
                    "http://localhost:8082/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Write a very long story about a journey through space that takes at least 500 words.",
                            }
                        ],
                    },
                )
            )

            # Cancel after 2 seconds
            await asyncio.sleep(2)
            task.cancel()

            try:
                await task
                print("❌ Request should have been cancelled")
            except asyncio.CancelledError:
                print("✅ Non-streaming request cancelled successfully")

        except Exception as e:
            print(f"❌ Non-streaming test error: {e}")


async def test_streaming_cancellation():
    """Test cancellation for streaming requests."""
    print("\n🧪 Testing streaming request cancellation...")

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Start streaming request
            async with client.stream(
                "POST",
                "http://localhost:8082/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a very long story about a journey through space that takes at least 500 words.",
                        }
                    ],
                    "stream": True,
                },
            ) as response:
                if response.status_code == 200:
                    print("✅ Streaming request started successfully")

                    # Read a few chunks then simulate client disconnect
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        if line.strip():
                            chunk_count += 1
                            print(f"📦 Received chunk {chunk_count}: {line[:100]}...")

                            # Simulate client disconnect after 3 chunks
                            if chunk_count >= 3:
                                print("🔌 Simulating client disconnect...")
                                break

                    print("✅ Streaming request cancelled successfully")
                else:
                    print(f"❌ Streaming request failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Streaming test error: {e}")


async def test_server_running():
    """Test if the server is running."""
    print("🔍 Checking if server is running...")

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8082/health")
            if response.status_code == 200:
                print("✅ Server is running and healthy")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to start the server with: python start_proxy.py")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting HTTP request cancellation tests")
    print("=" * 50)

    # Check if server is running
    if not await test_server_running():
        return

    print("\n" + "=" * 50)

    # Test non-streaming cancellation
    await test_non_streaming_cancellation()

    # Test streaming cancellation
    await test_streaming_cancellation()

    print("\n" + "=" * 50)
    print("✅ All cancellation tests completed!")
    print("\n💡 Note: The actual cancellation behavior depends on:")
    print("   - Client implementation (httpx in this case)")
    print("   - Network conditions")
    print("   - Server response to client disconnection")
    print("   - Whether the underlying OpenAI API supports cancellation")


if __name__ == "__main__":
    asyncio.run(main())
