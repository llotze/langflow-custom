"""
Configuration management for Flow Builder Agent.
Handles environment variables and API keys securely.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

# Load .env file from project root only
root_env = Path(__file__).parents[1] / '.env'
if root_env.exists():
    load_dotenv(root_env)
    logging.info(f"Loaded environment variables from {root_env}")
else:
    logging.warning("Root .env file not found. Using process environment variables only.")


class Config:
    """Configuration class for Flow Builder Agent."""
    
    # LLM Provider Configuration
    PROVIDER: str = os.getenv("FLOW_BUILDER_PROVIDER", "gemini")
    MODEL_NAME: str = os.getenv("FLOW_BUILDER_MODEL", "gemini-2.5-flash")
    
    # API Keys (read from root .env only)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Langflow Configuration
    LANGFLOW_API_URL: str = os.getenv("LANGFLOW_API_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:7860").rstrip('/'))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get the appropriate API key based on provider."""
        if cls.PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found in root .env. Please set it in the project .env file."
                )
            return cls.GEMINI_API_KEY
        elif cls.PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found in root .env. Please set it in the project .env file."
                )
            return cls.OPENAI_API_KEY
        else:
            raise ValueError(f"Unsupported provider: {cls.PROVIDER}")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        try:
            cls.get_api_key()
            return True
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def display_config(cls) -> str:
        """Return a safe string representation of the configuration."""
        def mask_key(key: Optional[str]) -> str:
            if not key:
                return "❌ Not Set"
            if len(key) < 8:
                return "***"
            return f"{key[:4]}...{key[-4:]}"
        
        return f"""
Flow Builder Agent Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provider: {cls.PROVIDER}
Model: {cls.MODEL_NAME}
Langflow API: {cls.LANGFLOW_API_URL}
Gemini API Key: {mask_key(cls.GEMINI_API_KEY)}
OpenAI API Key: {mask_key(cls.OPENAI_API_KEY)}
Log Level: {cls.LOG_LEVEL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
