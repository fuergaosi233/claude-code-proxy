#!/usr/bin/env python3
"""
Integration test script for HTTP request cancellation functionality.
These tests require a running server and are skipped in unit test runs.
"""

import asyncio
import os

import httpx
import pytest

# Skip these integration tests unless INTEGRATION_TEST env var is set
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("INTEGRATION_TEST"),
        reason="Integration tests require a running server. Set INTEGRATION_TEST=1 to run.",
    ),
]


async def test_server_running():
    """Test if the server is running."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("http://localhost:8082/health")
        assert response.status_code == 200


async def test_non_streaming_cancellation():
    """Test cancellation for non-streaming requests."""
    async with httpx.AsyncClient(timeout=30) as client:
        task = asyncio.create_task(
            client.post(
                "http://localhost:8082/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a very long story about a journey through space.",
                        }
                    ],
                },
            )
        )

        await asyncio.sleep(2)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


async def test_streaming_cancellation():
    """Test cancellation for streaming requests."""
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST",
            "http://localhost:8082/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a very long story about a journey through space.",
                    }
                ],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200

            chunk_count = 0
            async for line in response.aiter_lines():
                if line.strip():
                    chunk_count += 1
                    if chunk_count >= 3:
                        break

            # Successfully broke out of stream
            assert chunk_count >= 3
