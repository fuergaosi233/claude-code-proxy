"""Offline tests of the real route, converter, SDK and serialized HTTP boundary."""

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from openai import AsyncOpenAI

from src.api import endpoints
from src.conversion import request_converter
from src.core.client import OpenAIClient
from src.core.config import Config
from src.core.model_manager import ModelManager

REGIONS = ["https://api.minimax.io/v1", "https://api.minimaxi.com/v1/"]
REASONING = "Check the tool. 完整推理"
TEXT = "<think>Keep this legacy text.</think>"


@asynccontextmanager
async def proxy(monkeypatch, base_url=REGIONS[0], **settings):
    with monkeypatch.context() as patch:
        for name in ("BIG_MODEL", "MIDDLE_MODEL", "SMALL_MODEL", "AZURE_API_VERSION"):
            patch.delenv(name, raising=False)
        patch.setenv("OPENAI_API_KEY", "offline-fixture")
        patch.setenv("OPENAI_BASE_URL", base_url)
        for name, value in settings.items():
            patch.setenv(name, value)
        config = Config()
        captured = []

        def transport(request):
            body = json.loads(request.content)
            captured.append((str(request.url), body))
            message = {
                "role": "assistant",
                "reasoning_content": REASONING,
                "content": TEXT,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"value"}'},
                    }
                ],
            }
            if body.get("stream"):
                deltas = [
                    {"role": "assistant", "reasoning_content": "Check the tool. "},
                    {"reasoning_content": "完整推理"},
                    {"content": TEXT},
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_test",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            }
                        ]
                    },
                    {"tool_calls": [{"index": 0, "function": {"arguments": '"value"}'}}]},
                ]
                chunks = [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": body["model"],
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    for delta in deltas
                ]
                chunks.append(
                    {
                        **chunks[0],
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    }
                )
                chunks.append(
                    {
                        **chunks[0],
                        "choices": [],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
                    }
                )
                data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=data + "data: [DONE]\n\n",
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
                },
            )

        upstream = OpenAIClient(config.openai_api_key, config.openai_base_url)
        await upstream.client.close()
        upstream.client = AsyncOpenAI(
            api_key="offline-fixture",
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        )
        patch.setattr(endpoints, "config", config)
        patch.setattr(endpoints, "model_manager", ModelManager(config))
        patch.setattr(endpoints, "openai_client", upstream)
        patch.setattr(request_converter, "config", config)
        app = FastAPI()
        app.include_router(endpoints.router)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                yield client, captured
        finally:
            await upstream.client.close()


def request_body(model="claude-sonnet", stream=False, **extra):
    return {
        "model": model,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
        **extra,
    }


def stream_blocks(response):
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert not any(event["type"] == "error" for event in events)
    blocks = {}
    stopped = set()
    usage = None
    for event in events:
        if event["type"] == "content_block_start":
            assert event["index"] not in blocks
            blocks[event["index"]] = event["content_block"]
        elif event["type"] == "content_block_delta":
            assert event["index"] not in stopped
            block = blocks[event["index"]]
            delta = event["delta"]
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            elif delta["type"] == "input_json_delta":
                block["input"] = json.loads(delta["partial_json"])
        elif event["type"] == "content_block_stop":
            stopped.add(event["index"])
        elif event["type"] == "message_delta":
            usage = event["usage"]
    assert stopped == set(blocks)
    assert events[-1]["type"] == "message_stop"
    assert usage["output_tokens"] == 12
    return list(blocks.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", REGIONS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["claude-opus", "claude-sonnet", "claude-haiku"])
async def test_default_http_payload(monkeypatch, base_url, stream, model):
    async with proxy(monkeypatch, base_url) as (client, captured):
        response = await client.post("/v1/messages", json=request_body(model, stream))
        assert response.status_code == 200
        url, payload = captured[0]
        assert url == base_url.rstrip("/") + "/chat/completions"
        assert payload["model"] == "MiniMax-M3.1-Flash"
        assert payload["max_tokens"] == 200
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        for field in ("thinking", "reasoning_effort", "service_tier", "reasoning_split"):
            assert field not in payload
        if stream:
            assert payload["stream_options"] == {"include_usage": True}
            assert any(block["type"] == "thinking" for block in stream_blocks(response))


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("stream", [False, True])
async def test_flash_effort_http_payload(monkeypatch, effort, stream):
    async with proxy(monkeypatch) as (client, captured):
        response = await client.post(
            "/v1/messages",
            json=request_body(
                stream=stream, thinking={"type": "adaptive"}, output_config={"effort": effort}
            ),
        )
        assert response.status_code == 200
        payload = captured[0][1]
        assert payload["reasoning_effort"] == effort
        assert payload["thinking"] == {"type": "adaptive"}
        assert "extra_body" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"thinking": {"type": "disabled"}},
        {"thinking": {"enabled": False}},
        {"output_config": {"effort": "none"}},
        {"output_config": {"effort": "invalid"}},
        {"output_config": {"effort": {"invalid": True}}},
    ],
)
async def test_invalid_flash_controls_never_reach_transport(monkeypatch, extra):
    async with proxy(monkeypatch) as (client, captured):
        response = await client.post("/v1/messages", json=request_body(**extra))
        assert response.status_code == 400
        assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", REGIONS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["claude-sonnet", "MiniMax-M3"])
async def test_reasoning_and_tool_roundtrip(monkeypatch, base_url, stream, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this image"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
                },
            ],
        }
    ]
    tools = [{"name": "lookup", "input_schema": {"type": "object"}}]
    async with proxy(monkeypatch, base_url) as (client, captured):
        response = await client.post(
            "/v1/messages",
            json=request_body(
                model, stream=stream, messages=messages, system="Keep history", tools=tools
            ),
        )
        assert response.status_code == 200
        blocks = stream_blocks(response) if stream else response.json()["content"]
        assert (
            next(block["thinking"] for block in blocks if block["type"] == "thinking") == REASONING
        )
        messages += [
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_test", "content": "result"}
                ],
            },
        ]
        again = await client.post(
            "/v1/messages",
            json=request_body(
                model, stream=stream, messages=messages, system="Keep history", tools=tools
            ),
        )
        assert again.status_code == 200
        payload = captured[1][1]
        assert payload["messages"][0] == {"role": "system", "content": "Keep history"}
        assert payload["messages"][1]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        }
        assistant = payload["messages"][2]
        assert assistant["reasoning_content"] == REASONING
        assert assistant["content"] == TEXT
        assert assistant["tool_calls"] == [
            {
                "id": "call_test",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q": "value"}'},
            }
        ]
        assert payload["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call_test",
            "content": "result",
        }
        assert payload["tools"][0]["function"]["name"] == "lookup"


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", REGIONS)
@pytest.mark.parametrize(
    "thinking", [{"type": "disabled"}, {"type": "adaptive"}, {"enabled": False}]
)
async def test_m3_alternative(monkeypatch, base_url, thinking):
    async with proxy(monkeypatch, base_url) as (client, captured):
        response = await client.post(
            "/v1/messages",
            json=request_body("MiniMax-M3", thinking=thinking, output_config={"effort": "low"}),
        )
        assert response.status_code == 200
        payload = captured[0][1]
        assert payload["model"] == "MiniMax-M3"
        assert payload["thinking"] == {"type": thinking.get("type", "disabled")}
        assert "reasoning_effort" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["MiniMax-M2.7", "MiniMax-custom", "MiniMax-M3", "MiniMax-M3.1-Flash"]
)
async def test_explicit_model_ids(monkeypatch, model):
    async with proxy(monkeypatch) as (client, captured):
        response = await client.post("/v1/messages", json=request_body(model))
        assert response.status_code == 200
        assert captured[0][1]["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested,expected",
    [
        ("claude-opus", "MiniMax-M2.7"),
        ("claude-sonnet", "saved-custom"),
        ("claude-haiku", "another-custom"),
        ("another-custom", "another-custom"),
    ],
)
async def test_saved_and_custom_models(monkeypatch, requested, expected):
    async with proxy(
        monkeypatch,
        BIG_MODEL="MiniMax-M2.7",
        MIDDLE_MODEL="saved-custom",
        SMALL_MODEL="another-custom",
    ) as (client, captured):
        response = await client.post("/v1/messages", json=request_body(requested))
        assert response.status_code == 200
        assert captured[0][1]["model"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://gateway.example/v1",
        "https://api.minimax.io.example/v1",
    ],
)
async def test_other_endpoints_keep_existing_behavior(monkeypatch, base_url):
    async with proxy(monkeypatch, base_url) as (client, captured):
        response = await client.post(
            "/v1/messages",
            json=request_body(thinking={"type": "disabled"}, output_config={"effort": "none"}),
        )
        assert response.status_code == 200
        payload = captured[0][1]
        assert payload["model"] == "gpt-4o"
        assert "thinking" not in payload
        assert "reasoning_effort" not in payload
        assert not any(block["type"] == "thinking" for block in response.json()["content"])
