import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from fastapi import HTTPException


class TestOpenAIClientInit:
    """Tests for OpenAI client initialization."""

    def test_standard_client_init(self, monkeypatch):
        """Test standard OpenAI client initialization."""
        with patch("src.core.client.AsyncOpenAI") as mock_openai:
            from src.core.client import OpenAIClient

            client = OpenAIClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                timeout=90,
            )
            mock_openai.assert_called_once()
            assert client.api_key == "sk-test"
            assert client.base_url == "https://api.openai.com/v1"

    def test_azure_client_init(self, monkeypatch):
        """Test Azure OpenAI client initialization when api_version is provided."""
        with patch("src.core.client.AsyncAzureOpenAI") as mock_azure:
            from src.core.client import OpenAIClient

            client = OpenAIClient(
                api_key="azure-key",
                base_url="https://myinstance.openai.azure.com",
                timeout=60,
                api_version="2024-02-01",
            )
            mock_azure.assert_called_once()

    def test_custom_headers_merged(self, monkeypatch):
        """Test that custom headers are merged with default headers."""
        with patch("src.core.client.AsyncOpenAI") as mock_openai:
            from src.core.client import OpenAIClient

            client = OpenAIClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                custom_headers={"X-Custom": "value"},
            )
            call_kwargs = mock_openai.call_args[1]
            assert "X-Custom" in call_kwargs["default_headers"]
            assert call_kwargs["default_headers"]["X-Custom"] == "value"
            assert "Content-Type" in call_kwargs["default_headers"]


class TestClassifyOpenaiError:
    """Tests for error classification."""

    def _make_client(self):
        with patch("src.core.client.AsyncOpenAI"):
            from src.core.client import OpenAIClient

            return OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")

    def test_region_restriction(self):
        client = self._make_client()
        result = client.classify_openai_error("unsupported_country_region_territory detected")
        assert "region" in result.lower()

    def test_invalid_api_key(self):
        client = self._make_client()
        result = client.classify_openai_error("invalid_api_key")
        assert "API key" in result

    def test_rate_limit(self):
        client = self._make_client()
        result = client.classify_openai_error("rate_limit exceeded")
        assert "Rate limit" in result

    def test_model_not_found(self):
        client = self._make_client()
        result = client.classify_openai_error("model gpt-5 not found")
        assert "Model not found" in result

    def test_billing_issue(self):
        client = self._make_client()
        result = client.classify_openai_error("billing issue detected")
        assert "Billing" in result

    def test_unknown_error_returns_original(self):
        client = self._make_client()
        result = client.classify_openai_error("something weird happened")
        assert result == "something weird happened"


class TestCancelRequest:
    """Tests for request cancellation."""

    def _make_client(self):
        with patch("src.core.client.AsyncOpenAI"):
            from src.core.client import OpenAIClient

            return OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")

    def test_cancel_existing_request(self):
        client = self._make_client()
        event = asyncio.Event()
        client.active_requests["req-1"] = event
        assert client.cancel_request("req-1") is True
        assert event.is_set()

    def test_cancel_nonexistent_request(self):
        client = self._make_client()
        assert client.cancel_request("nonexistent") is False


class TestHandleOpenaiError:
    """Tests for _handle_openai_error method."""

    def _make_client(self):
        with patch("src.core.client.AsyncOpenAI"):
            from src.core.client import OpenAIClient

            return OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")

    def test_authentication_error(self):
        from unittest.mock import MagicMock

        from openai._exceptions import AuthenticationError

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "invalid_api_key"}}

        err = AuthenticationError(
            message="invalid_api_key",
            response=mock_response,
            body={"error": {"message": "invalid_api_key"}},
        )
        result = client._handle_openai_error(err)
        assert isinstance(result, HTTPException)
        assert result.status_code == 401

    def test_rate_limit_error(self):
        from unittest.mock import MagicMock

        from openai._exceptions import RateLimitError

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "rate_limit"}}

        err = RateLimitError(
            message="rate_limit",
            response=mock_response,
            body={"error": {"message": "rate_limit"}},
        )
        result = client._handle_openai_error(err)
        assert isinstance(result, HTTPException)
        assert result.status_code == 429

    def test_bad_request_error(self):
        from unittest.mock import MagicMock

        from openai._exceptions import BadRequestError

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "bad request"}}

        err = BadRequestError(
            message="bad request",
            response=mock_response,
            body={"error": {"message": "bad request"}},
        )
        result = client._handle_openai_error(err)
        assert isinstance(result, HTTPException)
        assert result.status_code == 400

    def test_generic_api_error(self):
        from unittest.mock import MagicMock

        from openai._exceptions import APIError

        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 503

        err = APIError(
            message="service unavailable",
            request=MagicMock(),
            body={"error": {"message": "service unavailable"}},
        )
        err.status_code = 503
        result = client._handle_openai_error(err)
        assert isinstance(result, HTTPException)
        assert result.status_code == 503

    def test_unexpected_error(self):
        client = self._make_client()
        err = RuntimeError("something went wrong")
        result = client._handle_openai_error(err)
        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert "Unexpected error" in result.detail


class TestCreateChatCompletion:
    """Tests for create_chat_completion."""

    @pytest.mark.asyncio
    async def test_successful_completion(self):
        """Test successful non-streaming completion."""
        with patch("src.core.client.AsyncOpenAI") as mock_openai_cls:
            from src.core.client import OpenAIClient

            mock_completion = MagicMock()
            mock_completion.model_dump.return_value = {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }

            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_openai_cls.return_value = mock_client_instance

            client = OpenAIClient(api_key="sk-test", base_url="https://api.openai.com/v1")

            result = await client.create_chat_completion(
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
            )

            assert result["id"] == "chatcmpl-test"
            assert result["choices"][0]["message"]["content"] == "Hello"
