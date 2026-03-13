import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _build_app():
    """Build a fresh FastAPI app with mocked dependencies."""
    from fastapi import FastAPI

    from src.api.endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


def _make_client_no_auth():
    """Create a TestClient that bypasses API key validation by patching config."""
    from src.core.config import Config

    # Create a config with no ANTHROPIC_API_KEY so validation is skipped
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
        cfg = Config.__new__(Config)
        cfg.openai_api_key = "sk-test-key-12345"
        cfg.anthropic_api_key = None
        cfg.openai_base_url = "https://api.openai.com/v1"
        cfg.azure_api_version = None
        cfg.host = "0.0.0.0"
        cfg.port = 8082
        cfg.log_level = "WARNING"
        cfg.max_tokens_limit = 4096
        cfg.min_tokens_limit = 100
        cfg.request_timeout = 90
        cfg.max_retries = 2
        cfg.big_model = "gpt-4o"
        cfg.middle_model = "gpt-4o"
        cfg.small_model = "gpt-4o-mini"

    return cfg


class TestHealthEndpoint:
    def test_health_check(self):
        app = _build_app()
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "openai_api_configured" in data
        assert "api_key_valid" in data

    def test_health_check_shows_api_configured(self):
        app = _build_app()
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["openai_api_configured"] is True


class TestRootEndpoint:
    def test_root_endpoint(self):
        app = _build_app()
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Claude-to-OpenAI API Proxy" in data["message"]
        assert data["status"] == "running"
        assert "endpoints" in data
        assert data["endpoints"]["messages"] == "/v1/messages"


class TestCountTokensEndpoint:
    def test_count_tokens_basic(self):
        cfg = _make_client_no_auth()
        with patch("src.api.endpoints.config", cfg):
            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello world"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "input_tokens" in data
            assert data["input_tokens"] > 0

    def test_count_tokens_with_system(self):
        cfg = _make_client_no_auth()
        with patch("src.api.endpoints.config", cfg):
            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages/count_tokens",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "system": "You are helpful.",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["input_tokens"] > 0


class TestApiKeyValidation:
    def test_rejects_invalid_key_when_anthropic_key_set(self, monkeypatch):
        """Test that invalid API key is rejected when ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "valid-key")

        from src.core.config import Config

        cfg = Config()

        with patch("src.api.endpoints.config", cfg):
            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"x-api-key": "wrong-key"},
            )
            assert response.status_code == 401

    def test_accepts_valid_key_when_anthropic_key_set(self, monkeypatch):
        """Test that valid API key is accepted."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "valid-key")

        from src.core.config import Config

        cfg = Config()

        with patch("src.api.endpoints.config", cfg), patch(
            "src.api.endpoints.openai_client"
        ) as mock_client:
            mock_client.create_chat_completion = AsyncMock(
                return_value={
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            )
            mock_client.classify_openai_error = MagicMock(return_value="error")

            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"x-api-key": "valid-key"},
            )
            assert response.status_code == 200

    def test_skips_validation_when_no_anthropic_key(self):
        """Test that validation is skipped when ANTHROPIC_API_KEY is not configured."""
        cfg = _make_client_no_auth()
        with patch("src.api.endpoints.config", cfg), patch(
            "src.api.endpoints.openai_client"
        ) as mock_client:
            mock_client.create_chat_completion = AsyncMock(
                return_value={
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            )
            mock_client.classify_openai_error = MagicMock(return_value="error")

            app = _build_app()
            client = TestClient(app)
            # No API key header
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 200


class TestCreateMessageEndpoint:
    def test_non_streaming_success(self):
        """Test successful non-streaming message creation."""
        cfg = _make_client_no_auth()
        with patch("src.api.endpoints.config", cfg), patch(
            "src.api.endpoints.openai_client"
        ) as mock_client:
            mock_client.create_chat_completion = AsyncMock(
                return_value={
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help?",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )
            mock_client.classify_openai_error = MagicMock(return_value="error")

            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert data["role"] == "assistant"
            assert data["content"][0]["text"] == "Hello! How can I help?"

    def test_streaming_response(self):
        """Test streaming message creation returns SSE stream."""
        cfg = _make_client_no_auth()

        async def mock_stream(*args, **kwargs):
            yield 'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}'
            yield 'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1}}'
            yield "data: [DONE]"

        with patch("src.api.endpoints.config", cfg), patch(
            "src.api.endpoints.openai_client"
        ) as mock_client:
            mock_client.create_chat_completion_stream = MagicMock(return_value=mock_stream())
            mock_client.cancel_request = MagicMock()
            mock_client.classify_openai_error = MagicMock(return_value="error")

            app = _build_app()
            client = TestClient(app)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
