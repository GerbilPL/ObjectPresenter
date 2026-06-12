import json
from pathlib import Path

class ConfigManager:
    """Manages application configuration and saves it to config.json"""

    def __init__(self, config_path: str = "config.json"):
        """Initializes ConfigManager with default configuration.
        
        Args:
            config_path: Path to the configuration JSON file (default: "config.json")
        """
        self.config_path = Path(config_path)
        self.default_config = {
            "theme": "System",
            "time_light_hour": "09:00",
            "time_dark_hour": "18:00",
            "filename_template": "filename$_extracted",
            "default_margin": 20,
            "margin_relative": False,
            "inpaint_enabled": False,
            "inpaint_method": "OpenCV",
            "batch_view_mode": "List",
            "device_preference": "Auto"
        }
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Loads configuration from JSON file with defaults fallback.
        
        Attempts to load config.json. If file missing or corrupted, silently
        returns default configuration (user loses custom settings).
        
        Returns:
            Dict with all configuration keys. Defaults merged with loaded values.
                Config keys: theme, time_light_hour, time_dark_hour, filename_template,
                default_margin, margin_relative, inpaint_enabled, inpaint_method,
                batch_view_mode, device_preference.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return {**self.default_config, **json.load(f)}
            except Exception as e:
                print(f"Failed to load config, using defaults: {e}")
        return self.default_config.copy()

    def save_config(self) -> None:
        """Persists current configuration to JSON file.
        
        Silently catches and prints errors if save fails.
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def get(self, key: str, default=None):
        """Retrieves a configuration value by key with optional default.
        
        Args:
            key: Configuration key (e.g., 'theme', 'device_preference').
            default: Value returned if key missing. Defaults to None.
        
        Returns:
            Configuration value for key, or default parameter if not found.
        """
        return self.config.get(key, default)

    def set(self, key: str, value) -> None:
        """Sets a configuration value and immediately saves to disk.
        
        Args:
            key: Configuration key (e.g., 'theme', 'device_preference').
            value: Any JSON-serializable value (str, int, bool, list, dict).
        
        Note:
            If save fails, config is still updated in memory but disk file
            not modified. Check console for error message.
        """
        self.config[key] = value
        self.save_config()