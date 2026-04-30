import json
from pathlib import Path

class ConfigManager:
    """Manages application configuration and saves it to config.json"""

    def __init__(self, config_path: str = "config.json"):
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
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return {**self.default_config, **json.load(f)}
            except Exception as e:
                print(f"Failed to load config, using defaults: {e}")
        return self.default_config.copy()

    def save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value) -> None:
        self.config[key] = value
        self.save_config()