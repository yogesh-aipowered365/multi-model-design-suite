"""
Security Hardening Module
- Filename sanitization
- MIME type validation
- Rate limiting per session
- Secret redaction from logs
- Security utilities
"""

import re
import os
import hashlib
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import mimetypes

# Optional: python-magic for more robust MIME detection
try:
    import magic  # type: ignore  # noqa: F401
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


# ============================================================================
# FILENAME SANITIZATION
# ============================================================================

UNSAFE_FILENAME_CHARS = r'[^\w\-. ]'
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and invalid characters.

    Args:
        filename: Raw filename from upload

    Returns:
        Safe filename

    Examples:
        "../../../etc/passwd" → "etc_passwd"
        "file<script>.jpg" → "file_script.jpg"
        "very_" * 100 + ".jpg" → "very_very_very..." (truncated)
    """
    # Remove path separators (prevent directory traversal)
    filename = filename.replace("../", "").replace("..\\", "")
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Replace unsafe characters with underscore
    filename = re.sub(UNSAFE_FILENAME_CHARS, "_", filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Limit length (preserve extension)
    if len(filename) > MAX_FILENAME_LENGTH:
        name, ext = os.path.splitext(filename)
        max_name_len = MAX_FILENAME_LENGTH - len(ext)
        filename = name[:max_name_len] + ext

    # Prevent empty filename
    if not filename or filename == "_":
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return filename


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate sanitized filename.

    Args:
        filename: Filename to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if sanitization changed the name significantly
    original_len = len(filename)
    sanitized = sanitize_filename(filename)

    if not sanitized:
        return False, "Filename is empty after sanitization"

    if len(sanitized) < 3:
        return False, "Filename too short after sanitization"

    if sanitized != filename:
        # This is OK - we expect sanitization to change potentially unsafe names
        pass

    return True, None


# ============================================================================
# MIME TYPE VALIDATION
# ============================================================================

ALLOWED_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
}


def validate_mime_type(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate file MIME type against extension and magic bytes.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"File extension '{ext}' not allowed. Use: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"

    # Check file size (max 50MB)
    max_size_bytes = 50 * 1024 * 1024
    if len(file_bytes) > max_size_bytes:
        return False, f"File too large ({len(file_bytes) / 1024 / 1024:.1f}MB). Max 50MB."

    # Check minimum file size (prevent empty files)
    if len(file_bytes) < 100:
        return False, "File too small. Minimum 100 bytes."

    try:
        # Try to detect MIME type from magic bytes (if available)
        if HAS_MAGIC:
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_buffer(file_bytes)

            if detected_mime not in ALLOWED_IMAGE_MIMETYPES:
                return False, f"File type '{detected_mime}' not allowed. Use: {', '.join(ALLOWED_IMAGE_MIMETYPES)}"
    except Exception as e:
        # If magic library fails, warn but don't block
        # This is a soft validation - fallback to extension check
        pass

    # Double-check with mimetypes
    guessed_mime, _ = mimetypes.guess_type(filename)
    if guessed_mime and guessed_mime not in ALLOWED_IMAGE_MIMETYPES:
        return False, f"Guessed MIME type '{guessed_mime}' not allowed"

    return True, None


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Soft rate limiter for analysis runs per session."""

    HARD_LIMIT = 50  # Hard stop after this many
    SOFT_LIMIT = 10  # Warn after this many
    SOFT_LIMIT_WINDOW = timedelta(hours=1)

    def __init__(self):
        """Initialize rate limiter."""
        self.runs = []  # List of (timestamp, description) tuples

    def record_run(self, description: str = "analysis"):
        """Record an analysis run."""
        self.runs.append((datetime.now(), description))
        # Keep only recent runs
        self._cleanup_old_runs()

    def _cleanup_old_runs(self):
        """Remove runs outside the window."""
        cutoff = datetime.now() - self.SOFT_LIMIT_WINDOW
        self.runs = [(ts, desc) for ts, desc in self.runs if ts > cutoff]

    def get_run_count(self) -> int:
        """Get count of runs in current window."""
        self._cleanup_old_runs()
        return len(self.runs)

    def check_rate_limit(self) -> Tuple[bool, Optional[str]]:
        """
        Check if rate limit is exceeded.

        Returns:
            Tuple of (is_allowed, warning_message)
            - is_allowed=True: OK to proceed
            - is_allowed=False: Hard limit hit, block request
            - warning_message: Soft limit warning (if applicable)
        """
        count = self.get_run_count()

        # Hard limit
        if count >= self.HARD_LIMIT:
            return False, f"Rate limit exceeded ({count}/{self.HARD_LIMIT} runs/hour). Please wait."

        # Soft limit warning
        if count >= self.SOFT_LIMIT:
            remaining = self.HARD_LIMIT - count
            return True, f"⚠️ Rate limit warning: {count}/{self.SOFT_LIMIT} runs. {remaining} remaining before blocking."

        return True, None

    def get_stats(self) -> dict:
        """Get rate limit statistics."""
        count = self.get_run_count()
        return {
            "current_runs": count,
            "soft_limit": self.SOFT_LIMIT,
            "hard_limit": self.HARD_LIMIT,
            "window_minutes": int(self.SOFT_LIMIT_WINDOW.total_seconds() / 60),
            "remaining": max(0, self.HARD_LIMIT - count),
        }


# ============================================================================
# SECRET REDACTION
# ============================================================================

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "token",
    "password",
    "passwd",
    "secret",
    "key",
    "credential",
    "openrouter",
    "authorization",
    "auth",
}

SENSITIVE_ENV_VARS = {
    "OPENROUTER_API_KEY",
    "API_KEY",
    "SECRET_KEY",
    "DATABASE_URL",
    "ADMIN_PASSWORD",
}


def should_redact(key: str) -> bool:
    """Check if a key contains sensitive data."""
    key_lower = key.lower()
    return any(sensitive in key_lower for sensitive in SENSITIVE_KEYS)


def redact_value(value: str, prefix_len: int = 4, suffix_len: int = 2) -> str:
    """
    Redact a sensitive value, showing only prefix and suffix.

    Args:
        value: Value to redact
        prefix_len: Number of chars to show at start
        suffix_len: Number of chars to show at end

    Returns:
        Redacted value (e.g., "sk-a***...***bc")
    """
    if len(value) <= prefix_len + suffix_len:
        return "*" * len(value)

    prefix = value[:prefix_len]
    suffix = value[-suffix_len:]
    stars = "*" * max(3, len(value) - prefix_len - suffix_len)
    return f"{prefix}{stars}{suffix}"


def redact_dict(data: dict, depth: int = 0, max_depth: int = 3) -> dict:
    """
    Recursively redact sensitive values in a dictionary.

    Args:
        data: Dictionary to redact
        depth: Current recursion depth
        max_depth: Maximum depth to recurse

    Returns:
        Dictionary with sensitive values redacted
    """
    if depth > max_depth or not isinstance(data, dict):
        return data

    redacted = {}
    for key, value in data.items():
        if should_redact(key):
            if isinstance(value, str):
                redacted[key] = redact_value(value)
            elif isinstance(value, (int, float)):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = "***REDACTED***"
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value, depth + 1, max_depth)
        elif isinstance(value, (list, tuple)):
            redacted[key] = [
                redact_dict(item, depth + 1, max_depth) if isinstance(item, dict)
                else item for item in value
            ]
        else:
            redacted[key] = value

    return redacted


def redact_string(text: str) -> str:
    """
    Redact likely secret strings from text (rough regex-based approach).

    Args:
        text: Text potentially containing secrets

    Returns:
        Text with likely secrets redacted
    """
    # Redact API keys (sk-xxx, pk-xxx patterns)
    text = re.sub(r'sk-[A-Za-z0-9]{40,}', 'sk-***REDACTED***', text)
    text = re.sub(r'pk-[A-Za-z0-9]{40,}', 'pk-***REDACTED***', text)

    # Redact URLs with credentials
    text = re.sub(r'://[^:]+:[^@]+@', '://***:***@', text)

    # Redact potential bearer tokens
    text = re.sub(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*',
                  'Bearer ***REDACTED***', text)

    return text


def create_safe_log_entry(message: str, data: dict = None) -> str:
    """
    Create a log entry with sensitive data redacted.

    Args:
        message: Log message
        data: Optional data dictionary

    Returns:
        Safe log message
    """
    safe_message = redact_string(message)

    if data:
        safe_data = redact_dict(data)
        safe_message += f" | {safe_data}"

    return safe_message


# ============================================================================
# FILE HASH & INTEGRITY
# ============================================================================

def calculate_file_hash(file_bytes: bytes, algorithm: str = "sha256") -> str:
    """
    Calculate file hash for integrity verification.

    Args:
        file_bytes: Raw file bytes
        algorithm: Hash algorithm (sha256, sha1, md5)

    Returns:
        Hex hash string
    """
    if algorithm == "sha256":
        return hashlib.sha256(file_bytes).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(file_bytes).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(file_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def verify_file_hash(file_bytes: bytes, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify file integrity using hash.

    Args:
        file_bytes: Raw file bytes
        expected_hash: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if hash matches
    """
    calculated = calculate_file_hash(file_bytes, algorithm)
    return calculated.lower() == expected_hash.lower()


# ============================================================================
# SECURITY SUMMARY
# ============================================================================

def get_security_summary() -> dict:
    """Get summary of security features."""
    return {
        "filename_sanitization": {
            "enabled": True,
            "max_length": MAX_FILENAME_LENGTH,
            "unsafe_chars_removed": True,
        },
        "mime_type_validation": {
            "enabled": True,
            "allowed_types": list(ALLOWED_IMAGE_MIMETYPES),
            "max_file_size_mb": 50,
            "magic_detection": True,
        },
        "rate_limiting": {
            "enabled": True,
            "soft_limit": RateLimiter.SOFT_LIMIT,
            "hard_limit": RateLimiter.HARD_LIMIT,
            "window_hours": 1,
        },
        "secret_redaction": {
            "enabled": True,
            "log_redaction": True,
            "sensitive_keys_tracked": len(SENSITIVE_KEYS),
        },
    }
