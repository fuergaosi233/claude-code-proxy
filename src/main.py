import asyncio
import logging
import sys

import uvicorn
from fastapi import FastAPI

from src.api.compat import router as compat_router
from src.api.endpoints import router as api_router
from src.core.config import config
from src.core.logging import log_active_models

_log = logging.getLogger("main")

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")

app.include_router(api_router)
# Compat router must come after the main router so explicit message-traffic
# routes win; its catch-all (registered last inside the router) then serves
# Anthropic-style 404s for any other /v1 or /api path.
app.include_router(compat_router)


@app.on_event("startup")
async def startup():
    """Start the background model health checker and run model discovery on server startup."""
    config.start_health_checker()

    # Run provider model discovery in the background so it doesn't block startup
    from src.core.provider_discovery import run_discovery_and_patch

    # We only probe the first provider in the rotation for limits to keep it simple
    if config.providers:
        prov = config.providers[0]
        asyncio.create_task(
            run_discovery_and_patch(
                base_url=prov["base_url"],
                api_key=prov["api_key"],
                custom_headers=config.get_custom_headers(),
            )
        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Claude-to-OpenAI API Proxy v1.0.0")
        print("")
        print("Usage: python src/main.py")
        print("")
        print("Required environment variables:")
        print("  OPENAI_API_KEY - Your OpenAI API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
        print("                      If set, clients must provide this exact API key")
        print("  OPENAI_BASE_URL - OpenAI API base URL (default: https://api.openai.com/v1)")
        print("  BIG_MODEL - Model for opus requests (default: gpt-4o)")
        print("  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)")
        print("  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
        print("  HOST - Server host (default: 0.0.0.0)")
        print("  PORT - Server port (default: 8082)")
        print("  LOG_LEVEL - Logging level (default: WARNING)")
        print("  MAX_TOKENS_LIMIT - Token limit (default: 4096)")
        print("  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)")
        print("  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet/opus models -> {config.big_model}")
        sys.exit(0)

    # Configuration summary - using logger
    _log.info("🚀 Claude-to-OpenAI API Proxy v1.0.0 starting")
    _log.info(
        "Configuration: BASE_URL=%s, Server=%s:%s", config.openai_base_url, config.host, config.port
    )
    _log.info(
        "Models - Big: %s, Middle: %s, Small: %s",
        config.big_model,
        config.middle_model,
        config.small_model,
    )
    _log.info(
        "Limits - Max tokens: %d, Min tokens: %d, Timeout: %ds, RPM: %d",
        config.max_tokens_limit,
        config.min_tokens_limit,
        config.request_timeout,
        config._rpm_limit,
    )
    _log.info(
        "Client API key validation: %s", "Enabled" if config.anthropic_api_key else "Disabled"
    )

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()

    # Validate and set default if invalid
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
