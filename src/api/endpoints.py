import os
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.core.client import OpenAIClient
from src.core.config import config
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest

router = APIRouter()

# Location of the per-request state file written for the Claude Code statusline.
# Mirrors the path resolution in src/core/logging.py:28-31 so this lands next to
# proxy.log regardless of the proxy's launch directory.
_endpoints_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_endpoints_file)))
_last_model_path = os.path.join(_project_root, "logs", ".last_model")


def _write_last_model(tier: str, model: str) -> None:
    """Persist the model the proxy actually routed this request to.

    The statusline script reads logs/.last_model to display the real upstream model
    (post health-aware fallback) rather than the model Claude Code requested. Writes
    are atomic (temp file + os.replace) so a concurrent statusline read never sees a
    partial file, and any failure is swallowed — the status file must never break a
    request.
    """
    try:
        log_dir = os.path.dirname(_last_model_path)
        if not os.path.isdir(log_dir):
            return
        tmp_path = f"{_last_model_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(f"{tier}\t{model}\n")
            fh.write(f"{datetime.now().isoformat()}\n")
        os.replace(tmp_path, _last_model_path)
    except Exception as exc:  # noqa: BLE001 — status file must never break a request
        logger.debug("Failed to write .last_model: %s", exc)


# Global client for compatibility; each request may obtain a fresh client via config.
openai_client = config.get_next_client()

# No global OpenAI client; each request obtains a client from config to enable round‑robin provider selection.


async def validate_api_key(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None

    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")

    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return

    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401, detail="Invalid API key. Please provide a valid Anthropic API key."
        )


@router.post("/v1/messages")
async def create_message(
    request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)
):
    try:
        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Map Claude model to OpenAI model and tier
        openai_model, model_tier = model_manager.map_claude_model_to_openai(request.model)

        # Log incoming request with model info
        logger.info(
            "Request [%s] model=%s -> %s (tier=%s, stream=%s)",
            request_id[:8],
            request.model,
            openai_model,
            model_tier,
            request.stream,
        )

        # Convert Claude request to OpenAI format
        openai_request = convert_claude_to_openai(request, model_manager)

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        # Get the model and tier from the model manager
        openai_model, model_tier = model_manager.map_claude_model_to_openai(request.model)
        # Update the OpenAI request with the resolved model
        openai_request["model"] = openai_model
        # Persist the actually-routed model (post-fallback) for the statusline.
        # Skip the "small" tier so the statusline keeps showing the last meaningful
        # (non-small) upstream model rather than the lightweight fallback.
        # Case-insensitive so "Small"/"SMALL" variants also skip the write.
        if (model_tier or "").lower() != "small":
            _write_last_model(model_tier or "unknown", openai_model)

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                client = await config.get_next_client_async()
                if client is None:
                    # All API keys exhausted - return 429
                    raise HTTPException(
                        status_code=429,
                        detail="All API keys exhausted. Please wait before trying again.",
                    )
                openai_stream = client.create_chat_completion_stream(
                    openai_request, request_id, model_tier
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except Exception as e:
                # Log unexpected errors (API errors are now yielded as events in the stream)
                logger.error(f"Unexpected streaming error: {e}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            # Non-streaming response
            client = config.get_next_client()
            try:
                openai_response = await client.create_chat_completion(
                    openai_request, request_id, model_tier
                )
                claude_response = convert_openai_to_claude_response(openai_response, request)
                return claude_response
            except HTTPException as e:
                # Re-raise HTTP exceptions as-is (client already reported the error)
                raise
            except Exception as e:
                # Log the error and re-raise (client already reported the error)
                logger.error(f"Non-streaming request failed: {e}")
                raise
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        # Reuse the same client if available, otherwise get a new one
        # (In practice, 'client' should be available from the try block above)
        # But to be safe, we'll try to get it from the failed request context
        error_message = "Internal server error"
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint with RPM info"""
    # Get current RPM stats
    now = time.time()
    rpm_info = []
    for api_key, usage in config._provider_usage.items():
        # Reset expired windows
        if now - usage["window_start"] >= 60:
            current_count = 0
        else:
            current_count = usage["count"]

        # Check penalty status
        penalty_remaining = 0
        if api_key in config._provider_penalties:
            penalty_remaining = max(0, config._provider_penalties[api_key] - now)

        rpm_info.append(
            {
                "key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key,
                "requests_this_minute": current_count,
                "rpm_limit": config._rpm_limit,
                "percentage_used": (
                    round((current_count / config._rpm_limit) * 100, 1)
                    if config._rpm_limit > 0
                    else 0
                ),
                "penalty_remaining_seconds": int(penalty_remaining),
                "window_age_seconds": int(now - usage["window_start"]),
            }
        )

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
        "rpm_info": rpm_info,
        "total_configured_keys": len(config.providers),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        client = config.get_next_client()
        test_response = await client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/v1/models")
async def list_models():
    """List available models in Anthropic and OpenAI dual-compatible format.

    Exposes exactly one alias per tier (opus/sonnet/haiku). The alias the client
    picks is irrelevant — model_manager routes it to the active model for that
    tier, as configured via BIG/MIDDLE/SMALL_MODEL in .env.
    """
    now_ts = int(time.time())
    now_iso = datetime.now().isoformat()

    # One representative per tier; .env tier mapping resolves the real model.
    claude_models = [
        {
            "id": "claude-3-opus-20240229",
            "display_name": "Opus (big tier)",
            "created_at": "2024-02-29T00:00:00Z",
            "created": 1709164800,
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "display_name": "Sonnet (middle tier)",
            "created_at": "2024-10-22T00:00:00Z",
            "created": 1729555200,
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "display_name": "Haiku (small tier)",
            "created_at": "2024-10-22T00:00:00Z",
            "created": 1729555200,
        },
    ]

    data = [
        {
            "id": m["id"],
            "type": "model",
            "object": "model",
            "display_name": m["display_name"],
            "created_at": m["created_at"],
            "created": m["created"],
            "owned_by": "claude-code-proxy",
        }
        for m in claude_models
    ]

    return {
        "object": "list",
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


@router.api_route("/api/hello", methods=["GET", "HEAD"])
@router.api_route("/hello", methods=["GET", "HEAD"])
async def api_hello():
    """Health probe endpoint used by Anthropic clients and Claude extensions."""
    return {"status": "ok"}


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "models": "/v1/models",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
