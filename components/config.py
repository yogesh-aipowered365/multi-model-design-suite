"""
Enhanced configuration management with validation and environment loading.
"""

import os
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
import json

load_dotenv()


class ConfigError(Exception):
    """Configuration error."""
    pass


class Config:
    """
    Centralized configuration for the application.
    Reads from environment variables with sensible defaults.
    """

    # ========================================================================
    # API & Model Configuration (CRITICAL)
    # ========================================================================
    # BYOK: API key is now provided by user at runtime, not from env
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/"
    )
    VISION_MODEL: str = os.getenv("VISION_MODEL", "openai/gpt-4o")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )

    # ========================================================================
    # Model Parameters (OPTIONAL)
    # ========================================================================

    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_K_RAG: int = int(os.getenv("TOP_K_RAG", "3"))

    # ========================================================================
    # Storage & Caching (OPTIONAL)
    # ========================================================================

    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")
    LOGS_DIR: str = os.path.join(STORAGE_DIR, "logs")
    CACHE_DIR: str = os.path.join(STORAGE_DIR, "cache")

    ENABLE_CACHE: bool = os.getenv(
        "ENABLE_CACHE", "true").lower() in ("true", "1", "yes")

    # ========================================================================
    # Analysis Storage Backend (OPTIONAL)
    # ========================================================================

    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "json").lower()
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "data/reports")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/reports.db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ========================================================================
    # Authentication (OPTIONAL)
    # ========================================================================

    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in (
        "true", "1", "yes")
    AUTH_USERNAME: str = os.getenv("AUTH_USERNAME", "admin")
    AUTH_PASSWORD: str = os.getenv("AUTH_PASSWORD", "password")
    # Format: "user1:pass1,user2:pass2"
    AUTH_USERS: str = os.getenv("AUTH_USERS", "")

    # ========================================================================
    # Logging (OPTIONAL)
    # ========================================================================

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.path.join(LOGS_DIR, "app.log")

    # ========================================================================
    # Application Metadata (OPTIONAL)
    # ========================================================================

    SITE_URL: str = os.getenv("SITE_URL", "http://localhost:8501")
    APP_NAME: str = os.getenv("APP_NAME", "DesignAnalysisPoc")

    @classmethod
    def validate(cls) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration at startup.

        Returns:
            Tuple of (is_valid, critical_errors, warnings)
            - is_valid (bool): True if all critical vars are set
            - critical_errors (List[str]): List of critical missing vars (blocks startup)
            - warnings (List[str]): List of optional var issues (info only)
        """
        critical_errors = []
        warnings = []

        # ====================================================================
        # CRITICAL: API Key (BYOK - provided at runtime in UI)
        # ====================================================================
        # Note: API key is now required to be provided by user at runtime via
        # the Streamlit UI. It's no longer required at startup since it comes
        # from session state. Environment variable is kept for backward compatibility.
        # if not cls.OPENROUTER_API_KEY or cls.OPENROUTER_API_KEY == "your-api-key-here":
        #     critical_errors.append(
        #         "❌ OPENROUTER_API_KEY not set. Get one from: https://openrouter.ai/keys"
        #     )

        # ====================================================================
        # CRITICAL: Model selection
        # ====================================================================
        if not cls.VISION_MODEL:
            critical_errors.append(
                "❌ VISION_MODEL not set. Recommended: openai/gpt-4o"
            )

        # ====================================================================
        # OPTIONAL: Parameter validation
        # ====================================================================
        if cls.MAX_TOKENS < 100 or cls.MAX_TOKENS > 10000:
            warnings.append(
                f"⚠️ MAX_TOKENS={cls.MAX_TOKENS} is unusual. Expected 100-10000."
            )

        if not (0.0 <= cls.TEMPERATURE <= 1.0):
            warnings.append(
                f"⚠️ TEMPERATURE={cls.TEMPERATURE} out of range (0-1)."
            )

        if cls.TOP_K_RAG < 1 or cls.TOP_K_RAG > 10:
            warnings.append(
                f"⚠️ TOP_K_RAG={cls.TOP_K_RAG} is unusual. Expected 1-10."
            )

        # ====================================================================
        # OPTIONAL: Storage directory
        # ====================================================================
        try:
            os.makedirs(cls.STORAGE_DIR, exist_ok=True)
            os.makedirs(cls.LOGS_DIR, exist_ok=True)
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
        except OSError as e:
            warnings.append(
                f"⚠️ Could not create storage directory: {cls.STORAGE_DIR} ({e})"
            )

        # ====================================================================
        # OPTIONAL: Log level
        # ====================================================================
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if cls.LOG_LEVEL not in valid_levels:
            warnings.append(
                f"⚠️ Invalid LOG_LEVEL={cls.LOG_LEVEL}. Using INFO."
            )
            cls.LOG_LEVEL = "INFO"

        is_valid = len(critical_errors) == 0

        return is_valid, critical_errors, warnings

    @classmethod
    def get_summary(cls) -> Dict[str, any]:
        """Get config summary for logging."""
        return {
            "VISION_MODEL": cls.VISION_MODEL,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "MAX_TOKENS": cls.MAX_TOKENS,
            "TEMPERATURE": cls.TEMPERATURE,
            "TOP_K_RAG": cls.TOP_K_RAG,
            "ENABLE_CACHE": cls.ENABLE_CACHE,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "STORAGE_DIR": cls.STORAGE_DIR,
        }
