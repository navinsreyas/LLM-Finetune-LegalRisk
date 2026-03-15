"""
Centralized configuration and API key management.

Loads API keys from .env file so they are never hardcoded in source code.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError(
        "python-dotenv is required. Install with: pip install python-dotenv"
    )


def load_config() -> dict:
    """
    Load configuration from .env file.
    
    Looks for .env in the project root directory.
    Returns a dict with all config values.
    """
    # Find .env file — walk up from this file's location to find project root
    current = Path(__file__).resolve()
    for parent in [current.parent, current.parent.parent, current.parent.parent.parent]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        # Also try CWD
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)
    
    return {
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "hf_token": os.environ.get("HF_TOKEN"),
        "wandb_project": os.environ.get("WANDB_PROJECT", "legalrisk-llm"),
    }


def get_gemini_key() -> str:
    """Get Gemini API key or raise a helpful error."""
    config = load_config()
    key = config["gemini_api_key"]
    
    if not key or key == "your_gemini_api_key_here":
        raise ValueError(
            "\n" + "=" * 60 + "\n"
            "  GEMINI_API_KEY not set!\n\n"
            "  1. Get a FREE key at: https://aistudio.google.com/apikey\n"
            "  2. Open the .env file in your project root\n"
            "  3. Replace 'your_gemini_api_key_here' with your actual key\n"
            "=" * 60
        )
    
    return key


def get_anthropic_key() -> str:
    """Get Anthropic API key or raise a helpful error."""
    config = load_config()
    key = config["anthropic_api_key"]
    
    if not key or key == "your_anthropic_key_here":
        raise ValueError(
            "\n" + "=" * 60 + "\n"
            "  ANTHROPIC_API_KEY not set!\n\n"
            "  1. Get a key at: https://console.anthropic.com\n"
            "  2. Open the .env file in your project root\n"
            "  3. Add: ANTHROPIC_API_KEY=your_key\n"
            "=" * 60
        )
    
    return key
