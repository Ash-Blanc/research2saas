"""Centralized configuration for research2saas platform"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM API Keys
    mistral_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Pollinations.ai
    pollinations_api_key: str = ""
    pollinations_base_url: str = "https://gen.pollinations.ai/v1"
    pollinations_model: str = "nova-fast"  # openai, mistral, gemini, deepseek, etc.
    
    # Semantic Scholar
    s2_api_key: str = ""
    s2_base_url: str = "https://api.semanticscholar.org/graph/v1"
    s2_recommendations_url: str = "https://api.semanticscholar.org/recommendations/v1"
    s2_requests_per_second: float = 0.33  # Free tier
    s2_search_rate_limit: float = 0.2
    s2_cache_ttl: int = 3600  # 1 hour
    s2_cache_maxsize: int = 1000
    
    # HTTP Timeouts
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    
    # Database (optional)
    database_url: str = ""
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience accessors
def get_mistral_api_key() -> str:
    return get_settings().mistral_api_key


def get_s2_api_key() -> Optional[str]:
    key = get_settings().s2_api_key
    return key if key else None


def get_pollinations_api_key() -> Optional[str]:
    """Get Pollinations.ai API key"""
    key = get_settings().pollinations_api_key
    return key if key else None
