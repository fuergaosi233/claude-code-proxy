from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from src.api.endpoints import router as api_router
from src.web.routes import router as web_router
import uvicorn
from src.core.config import config, config_manager

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.1.0")

# API Router
app.include_router(api_router)

# Web UI Router (for configuration)
app.include_router(web_router)


def print_startup_message():
    """Prints a summary of the initial configuration on startup."""
    print("🚀 Claude-to-OpenAI API Proxy v1.1.0")
    print(f"✅ Configuration profile '{config_manager.get_active_profile_name()}' loaded.")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Server running at: http://{config.host}:{config.port}")
    print(f"   Configuration UI: http://{config.host}:{config.port}/")
    print(f"   Client API Key Validation: {'Enabled' if config.anthropic_api_key else 'Disabled'}")
    print("")


def main():
    print_startup_message()

    # Parse log level
    log_level = config.log_level.lower()
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    if log_level not in valid_levels:
        log_level = 'info'

    # Start server
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=log_level,
        reload=False,  # Set to True for development to auto-reload on code changes
    )


if __name__ == "__main__":
    main()
