import os
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for src.core.config.Config."""

    def test_config_loads_with_valid_env(self, monkeypatch):
        """Test config loads successfully with required env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api.com/v1")
        monkeypatch.setenv("BIG_MODEL", "gpt-4o")
        monkeypatch.setenv("SMALL_MODEL", "gpt-4o-mini")

        from src.core.config import Config

        cfg = Config()
        assert cfg.openai_api_key == "sk-test-key"
        assert cfg.openai_base_url == "https://custom.api.com/v1"
        assert cfg.big_model == "gpt-4o"
        assert cfg.small_model == "gpt-4o-mini"

    def test_config_missing_api_key_raises(self, monkeypatch):
        """Test that missing OPENAI_API_KEY raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from src.core.config import Config

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Config()

    def test_config_default_values(self, monkeypatch):
        """Test that default values are used when env vars are not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        # Remove optional env vars
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("BIG_MODEL", raising=False)
        monkeypatch.delenv("SMALL_MODEL", raising=False)
        monkeypatch.delenv("MAX_TOKENS_LIMIT", raising=False)
        monkeypatch.delenv("MIN_TOKENS_LIMIT", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        monkeypatch.delenv("HOST", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("MIDDLE_MODEL", raising=False)

        from src.core.config import Config

        cfg = Config()
        assert cfg.openai_base_url == "https://api.openai.com/v1"
        assert cfg.big_model == "gpt-4o"
        assert cfg.small_model == "gpt-4o-mini"
        assert cfg.max_tokens_limit == 4096
        assert cfg.min_tokens_limit == 100
        assert cfg.port == 8082
        assert cfg.host == "0.0.0.0"
        assert cfg.log_level == "INFO"

    def test_middle_model_defaults_to_big_model(self, monkeypatch):
        """Test that MIDDLE_MODEL defaults to BIG_MODEL value."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("BIG_MODEL", "custom-big-model")
        monkeypatch.delenv("MIDDLE_MODEL", raising=False)

        from src.core.config import Config

        cfg = Config()
        assert cfg.middle_model == "custom-big-model"

    def test_middle_model_custom_value(self, monkeypatch):
        """Test that MIDDLE_MODEL can be set independently."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("BIG_MODEL", "gpt-4o")
        monkeypatch.setenv("MIDDLE_MODEL", "gpt-4o-middle")

        from src.core.config import Config

        cfg = Config()
        assert cfg.middle_model == "gpt-4o-middle"

    def test_anthropic_api_key_validation_disabled_when_not_set(self, monkeypatch):
        """Test that client API key validation is skipped when ANTHROPIC_API_KEY not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from src.core.config import Config

        cfg = Config()
        assert cfg.anthropic_api_key is None
        assert cfg.validate_client_api_key("any-key") is True

    def test_anthropic_api_key_validation_enabled(self, monkeypatch):
        """Test client API key validation when ANTHROPIC_API_KEY is set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")

        from src.core.config import Config

        cfg = Config()
        assert cfg.validate_client_api_key("secret-key") is True
        assert cfg.validate_client_api_key("wrong-key") is False

    def test_custom_headers_parsing(self, monkeypatch):
        """Test custom headers are parsed from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("CUSTOM_HEADER_X_CUSTOM_AUTH", "bearer-token")
        monkeypatch.setenv("CUSTOM_HEADER_ANOTHER_HEADER", "value")

        from src.core.config import Config

        cfg = Config()
        headers = cfg.get_custom_headers()
        assert "X-CUSTOM-AUTH" in headers
        assert headers["X-CUSTOM-AUTH"] == "bearer-token"
        assert "ANOTHER-HEADER" in headers
        assert headers["ANOTHER-HEADER"] == "value"

    def test_validate_api_key_non_empty(self, monkeypatch):
        """Test validate_api_key returns True for non-empty keys (relaxed check)."""
        monkeypatch.setenv("OPENAI_API_KEY", "some-custom-key-format")

        from src.core.config import Config

        cfg = Config()
        assert cfg.validate_api_key() is True

    def test_validate_api_key_with_sk_prefix(self, monkeypatch):
        """Test validate_api_key works with standard sk- prefix."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard-key")

        from src.core.config import Config

        cfg = Config()
        assert cfg.validate_api_key() is True

    def test_azure_api_version(self, monkeypatch):
        """Test Azure API version configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("AZURE_API_VERSION", "2024-02-01")

        from src.core.config import Config

        cfg = Config()
        assert cfg.azure_api_version == "2024-02-01"
