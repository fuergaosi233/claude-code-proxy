from src.core.config import config


class ModelManager:
    def __init__(self, config):
        self.config = config

    def map_claude_model_to_openai(self, claude_model: str) -> tuple[str, str]:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern.

        Returns a tuple of (model_name, tier) where tier is one of 'big', 'middle', 'small'.
        The model_name is the currently active model for that tier (respecting fallbacks).
        """
        # If it's already an OpenAI model, return as-is
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model, "unknown"

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (
            claude_model.startswith("ep-")
            or claude_model.startswith("doubao-")
            or claude_model.startswith("deepseek-")
        ):
            return claude_model, "unknown"

        # Map based on model naming patterns, using active model for the tier
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.config.get_active_model("small"), "small"
        elif "sonnet" in model_lower:
            return self.config.get_active_model("middle"), "middle"
        elif "opus" in model_lower:
            return self.config.get_active_model("big"), "big"
        else:
            # Default to big model for unknown models
            return self.config.get_active_model("big"), "big"


model_manager = ModelManager(config)
