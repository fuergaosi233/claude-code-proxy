import os
import json
from typing import Dict, Any, Optional

class Config:
    """Holds the application configuration."""
    def __init__(self):
        # Load all settings from environment variables first as a base default
        self.host: str = os.environ.get("HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("PORT", 8082))
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO")
        
        # Load dynamic settings from environment as the base layer
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", 4096))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", 100))
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", 90))
        self.max_retries = int(os.environ.get("MAX_RETRIES", 2))

    def update(self, data: Dict[str, Any]):
        """
        Updates the configuration from a dictionary (a profile).
        It only updates if a key is present in the dictionary.
        """
        if "OPENAI_API_KEY" in data: self.openai_api_key = data["OPENAI_API_KEY"]
        if "ANTHROPIC_API_KEY" in data: self.anthropic_api_key = data["ANTHROPIC_API_KEY"]
        if "OPENAI_BASE_URL" in data: self.openai_base_url = data["OPENAI_BASE_URL"]
        if "AZURE_API_VERSION" in data: self.azure_api_version = data["AZURE_API_VERSION"]
        if "BIG_MODEL" in data: self.big_model = data["BIG_MODEL"]
        if "MIDDLE_MODEL" in data: self.middle_model = data.get("BIG_MODEL") # Fallback
        if "SMALL_MODEL" in data: self.small_model = data["SMALL_MODEL"]
        if "MAX_TOKENS_LIMIT" in data: self.max_tokens_limit = int(data["MAX_TOKENS_LIMIT"])
        if "MIN_TOKENS_LIMIT" in data: self.min_tokens_limit = int(data["MIN_TOKENS_LIMIT"])
        if "REQUEST_TIMEOUT" in data: self.request_timeout = int(data["REQUEST_TIMEOUT"])
        if "MAX_RETRIES" in data: self.max_retries = int(data["MAX_RETRIES"])

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dynamic parts of the config to a dictionary."""
        return {
            "OPENAI_API_KEY": self.openai_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "OPENAI_BASE_URL": self.openai_base_url,
            "AZURE_API_VERSION": self.azure_api_version,
            "BIG_MODEL": self.big_model,
            "MIDDLE_MODEL": self.middle_model,
            "SMALL_MODEL": self.small_model,
            "MAX_TOKENS_LIMIT": self.max_tokens_limit,
            "MIN_TOKENS_LIMIT": self.min_tokens_limit,
            "REQUEST_TIMEOUT": self.request_timeout,
            "MAX_RETRIES": self.max_retries,
        }

    def validate_client_api_key(self, client_api_key: str) -> bool:
        """Validate client's Anthropic API key."""
        if not self.anthropic_api_key:
            return True
        return client_api_key == self.anthropic_api_key


class ConfigManager:
    """Manages loading, saving, and switching configuration profiles."""
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.profiles_path = os.path.join(self.config_dir, "profiles.json")
        self.data: Dict[str, Any] = {"active_profile": "default", "profiles": {}}
        self.config = Config()  # Initialize the single config object
        self._ensure_config_exists()
        self.load_config()

    def _ensure_config_exists(self):
        """Ensures the config directory and default profiles.json exist."""
        os.makedirs(self.config_dir, exist_ok=True)
        if not os.path.exists(self.profiles_path):
            # Create a default profile from the current environment variables.
            # A temporary Config object will be initialized with all values from the env.
            temp_config = Config()
            self.data["profiles"]["default"] = temp_config.to_dict()
            self.save_profiles()

    def load_config(self):
        """
        Loads the active profile, layering it on top of the base config from env vars.
        """
        # The self.config object is already initialized with env vars.
        # Now, we load the JSON profiles and apply the active one on top.
        with open(self.profiles_path, "r") as f:
            self.data = json.load(f)
        
        active_profile_name = self.data.get("active_profile", "default")
        profile_data = self.data["profiles"].get(active_profile_name, {})
        
        # Layer the profile data on top of the base config
        self.config.update(profile_data)
        
        print(f"✅ Configuration updated for profile: '{active_profile_name}'")

    def save_profiles(self):
        """Saves the current state of profiles to the JSON file."""
        with open(self.profiles_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_active_profile_name(self) -> str:
        return self.data.get("active_profile", "default")

    def get_all_profiles(self) -> Dict[str, Any]:
        return self.data.get("profiles", {})

    def save_profile(self, name: str, profile_data: Dict[str, Any]):
        """Saves a profile and reloads the config if it's the active one."""
        # Ensure all fields are present by doing a round-trip with the Config object
        temp_config = Config()
        temp_config.update(profile_data)
        self.data["profiles"][name] = temp_config.to_dict()
        
        self.save_profiles()
        if name == self.get_active_profile_name():
            self.load_config()

    def rename_profile(self, old_name: str, new_name: str):
        """Renames a profile."""
        if old_name not in self.data["profiles"]:
            raise ValueError(f"Profile '{old_name}' not found.")
        if new_name in self.data["profiles"]:
            raise ValueError(f"Profile '{new_name}' already exists.")
        if not new_name.strip():
            raise ValueError("New profile name cannot be empty.")
        if old_name == "default":
            raise ValueError("Cannot rename the default profile.")

        # Rename the profile
        self.data["profiles"][new_name] = self.data["profiles"].pop(old_name)

        # If the renamed profile was active, update the active profile name
        if self.data.get("active_profile") == old_name:
            self.data["active_profile"] = new_name
        
        self.save_profiles()

    def activate_profile(self, name: str):
        """Sets a profile as active and reloads the configuration."""
        if name not in self.data["profiles"]:
            raise ValueError(f"Profile '{name}' not found.")
        self.data["active_profile"] = name
        self.save_profiles()
        self.load_config()

    def delete_profile(self, name: str):
        """Deletes a profile."""
        if name not in self.data["profiles"]:
            raise ValueError(f"Profile '{name}' not found.")
        if name == "default":
            raise ValueError("Cannot delete the default profile.")
        if name == self.get_active_profile_name():
            raise ValueError("Cannot delete the active profile. Switch to another profile first.")
            
        del self.data["profiles"][name]
        self.save_profiles()

# Global instance of the ConfigManager
config_manager = ConfigManager()
config = config_manager.config
