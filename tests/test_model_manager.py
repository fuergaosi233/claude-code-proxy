from unittest.mock import MagicMock

import pytest


class TestModelManager:
    """Tests for src.core.model_manager.ModelManager."""

    def _make_manager(self, big="gpt-4o", middle="gpt-4o", small="gpt-4o-mini"):
        from src.core.model_manager import ModelManager

        config = MagicMock()
        config.big_model = big
        config.middle_model = middle
        config.small_model = small
        return ModelManager(config)

    def test_haiku_maps_to_small_model(self):
        manager = self._make_manager(small="gpt-4o-mini")
        assert manager.map_claude_model_to_openai("claude-3-haiku-20240307") == "gpt-4o-mini"

    def test_sonnet_maps_to_middle_model(self):
        manager = self._make_manager(middle="gpt-4o-middle")
        assert manager.map_claude_model_to_openai("claude-3-5-sonnet-20241022") == "gpt-4o-middle"

    def test_opus_maps_to_big_model(self):
        manager = self._make_manager(big="gpt-4o-big")
        assert manager.map_claude_model_to_openai("claude-3-opus-20240229") == "gpt-4o-big"

    def test_gpt_model_passthrough(self):
        manager = self._make_manager()
        assert manager.map_claude_model_to_openai("gpt-4o") == "gpt-4o"
        assert manager.map_claude_model_to_openai("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_o1_model_passthrough(self):
        manager = self._make_manager()
        assert manager.map_claude_model_to_openai("o1-preview") == "o1-preview"

    def test_deepseek_model_passthrough(self):
        manager = self._make_manager()
        assert manager.map_claude_model_to_openai("deepseek-chat") == "deepseek-chat"

    def test_doubao_model_passthrough(self):
        manager = self._make_manager()
        assert manager.map_claude_model_to_openai("doubao-pro-32k") == "doubao-pro-32k"

    def test_ep_prefix_passthrough(self):
        manager = self._make_manager()
        assert manager.map_claude_model_to_openai("ep-20240101-abc") == "ep-20240101-abc"

    def test_unknown_model_defaults_to_big(self):
        manager = self._make_manager(big="my-big-model")
        assert manager.map_claude_model_to_openai("some-unknown-model") == "my-big-model"

    def test_case_insensitive_mapping(self):
        manager = self._make_manager(small="gpt-4o-mini", middle="gpt-4o", big="gpt-4o-big")
        assert manager.map_claude_model_to_openai("Claude-3-Haiku-Latest") == "gpt-4o-mini"
        assert manager.map_claude_model_to_openai("Claude-3-5-SONNET-Latest") == "gpt-4o"
        assert manager.map_claude_model_to_openai("Claude-3-OPUS-Latest") == "gpt-4o-big"
