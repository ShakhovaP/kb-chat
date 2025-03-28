import os

class ConfigManager:
    """
    Centralized configuration management for knowledge base service.
    """
    @staticmethod
    def get_required_env(key: str) -> str:
        """
        Retrieve and validate environment variables.
        
        :param key: Environment variable name
        :return: Environment variable value
        :raises ValueError: If environment variable is not set
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value