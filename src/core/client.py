import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import APIError, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.core.config import config

_log = logging.getLogger("client")


class OpenAIClient:
    """Async OpenAI client with cancellation support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 90,
        api_version: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.api_version = api_version
        self.custom_headers = custom_headers or {}
        self._last_model: Optional[str] = None
        self._last_model_tier: Optional[str] = None
        self.active_requests: Dict[str, asyncio.Event] = {}

        # Prepare default headers
        default_headers = {"Content-Type": "application/json", "User-Agent": "claude-proxy/1.0.0"}

        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}

        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers,
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout, default_headers=all_headers
            )

    def _track_model(self, request: Dict[str, Any], tier: Optional[str] = None):
        """Record the model being used for this request."""
        self._last_model = request.get("model")
        self._last_model_tier = tier

    async def create_chat_completion(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
        model_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        self._track_model(request, model_tier)

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Create task that can be cancelled
            completion_task = asyncio.create_task(self.client.chat.completions.create(**request))

            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")

                completion = await completion_task
            else:
                completion = await completion_task

            # Convert to dict format that matches the original interface
            result = completion.model_dump()

            # Report successful model usage
            self.report_model_success()

            return result

        except AuthenticationError as e:
            self._report_model_error(401, str(e))
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            self._report_model_error(429, str(e))
            config.report_rate_limit(self.api_key)
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))

        except BadRequestError as e:
            self._report_model_error(400, str(e))
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, "status_code", 500)
            self._report_model_error(status_code, str(e))
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            self._report_model_error(500, str(e))
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(
        self,
        request: Dict[str, Any],
        request_id: Optional[str] = None,
        model_tier: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        self._track_model(request, model_tier)

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True

            # Create the streaming completion
            streaming_completion = await self.client.chat.completions.create(**request)

            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        # Send cancellation event instead of raising exception
                        yield f'data: {{"error": {{"message": "Request cancelled by client", "type": "cancelled"}}}}'
                        self._report_model_error(499, "Request cancelled")
                        return

                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            # Signal end of stream
            yield "data: [DONE]"

            # Report successful model usage (only if we got here without errors)
            self.report_model_success()

        except AuthenticationError as e:
            self._report_model_error(401, str(e))
            error_msg = self.classify_openai_error(str(e))
            yield f'data: {{"error": {{"message": "{error_msg}", "type": "authentication_error"}}}}'
        except RateLimitError as e:
            self._report_model_error(429, str(e))
            config.report_rate_limit(self.api_key)
            error_msg = self.classify_openai_error(str(e))
            yield f'data: {{"error": {{"message": "{error_msg}", "type": "rate_limit_error"}}}}'

        except BadRequestError as e:
            self._report_model_error(400, str(e))
            error_msg = self.classify_openai_error(str(e))
            yield f'data: {{"error": {{"message": "{error_msg}", "type": "bad_request_error"}}}}'
        except APIError as e:
            status_code = getattr(e, "status_code", 500)
            self._report_model_error(status_code, str(e))
            error_msg = self.classify_openai_error(str(e))
            yield f'data: {{"error": {{"message": "{error_msg}", "type": "api_error"}}}}'
        except Exception as e:
            self._report_model_error(500, str(e))
            yield f'data: {{"error": {{"message": "Unexpected error: {str(e)}", "type": "internal_error"}}}}'

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()

        # Context length / token limit exceeded
        if (
            "context_length_exceeded" in error_str
            or "maximum context length" in error_str
            or "string too long" in error_str
            or "context window" in error_str
            or "too many tokens" in error_str
        ):
            return (
                "Context window exceeded: the conversation is too long for this model. "
                "The proxy attempted to trim the history automatically. "
                "If this persists, consider increasing CONTEXT_TRIM_RATIO or using a model "
                "with a larger context window."
            )

        # Region/country restrictions
        if (
            "unsupported_country_region_territory" in error_str
            or "country, region, or territory not supported" in error_str
        ):
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."

        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."

        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."

        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."

        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        # Default: return original message
        return str(error_detail)

    def _report_model_error(self, status_code: int, error_detail: str):
        """Report model errors to config for fallback tracking."""
        if not self._last_model or not self._last_model_tier:
            return
        if self._last_model_tier == "unknown":
            return

        error_str = str(error_detail).lower()
        is_model_error = (
            status_code == 429
            or status_code == 503
            or "rate_limit" in error_str
            or "model" in error_str
            and (
                "not found" in error_str
                or "does not exist" in error_str
                or "unavailable" in error_str
            )
            or "overloaded" in error_str
            or "capacity" in error_str
        )

        # Log the error
        _log.warning(
            "Model error [%s] %s: %s (status=%d, model_error=%s)",
            self._last_model_tier.upper(),
            self._last_model,
            str(error_detail)[:100],
            status_code,
            is_model_error,
        )

        if is_model_error:
            config.report_model_failure(self._last_model, self._last_model_tier)

    def report_model_success(self):
        """Report a successful model request."""
        if self._last_model and self._last_model_tier and self._last_model_tier != "unknown":
            _log.debug("Model success [%s] %s", self._last_model_tier.upper(), self._last_model)
            config.report_model_success(self._last_model, self._last_model_tier)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
