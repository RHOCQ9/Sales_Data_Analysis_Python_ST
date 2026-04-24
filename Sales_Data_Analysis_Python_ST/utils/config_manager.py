import json
import os
import streamlit as st

class ConfigManager:
    def __init__(self, file_path="favorites.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({}, f)

    def save_config(self, name, config_dict):
        """Guarda una configuración con un nombre dado."""
        configs = self.load_all_configs()
        configs[name] = config_dict
        with open(self.file_path, 'w') as f:
            json.dump(configs, f)

    def load_all_configs(self):
        """Carga todas las configuraciones guardadas."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def apply_config(self, name):
        """Aplica una configuración guardada al session_state de Streamlit."""
        configs = self.load_all_configs()
        if name in configs:
            config = configs[name]
            for key, value in config.items():
                st.session_state[key] = value
            return True
        return False
