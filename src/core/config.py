import os
import sys
from typing import List

# Configuration
class Config:
    def __init__(self):
        openai_api_key_str = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key_str:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Support multiple API keys separated by commas
        self.openai_api_keys = [key.strip() for key in openai_api_key_str.split(",") if key.strip()]
        if not self.openai_api_keys:
            raise ValueError("No valid OPENAI_API_KEY found in environment variables")
        
        # Keep backward compatibility - first key as primary
        self.openai_api_key = self.openai_api_keys[0]
        
        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")
        
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))
        
        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        
        # Model settings - BIG and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
        
    def validate_api_key(self):
        """Basic API key validation for all keys"""
        if not self.openai_api_keys:
            return False
        # Basic format check for OpenAI API keys
        for key in self.openai_api_keys:
            if not key.startswith('sk-'):
                return False
        return True
    
    def get_api_key_count(self):
        """Get the number of configured API keys"""
        return len(self.openai_api_keys)
        
    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in the environment, skip validation
        if not self.anthropic_api_key:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

try:
    config = Config()
    key_count = config.get_api_key_count()
    print(f" Configuration loaded: {key_count} API_KEY(s) configured, BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
