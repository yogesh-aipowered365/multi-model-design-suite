"""
Authentication module for Design Analysis application.
Supports optional authentication with environment-based credentials.
"""

import os
import hashlib
from typing import Optional, Tuple
import streamlit as st

from components.logging_utils import get_logger

logger = get_logger(__name__)


class AuthConfig:
    """Authentication configuration."""

    # Check if auth is enabled
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Default credentials (for demo/dev)
    DEFAULT_USERNAME: str = os.getenv("AUTH_USERNAME", "admin")
    DEFAULT_PASSWORD: str = os.getenv("AUTH_PASSWORD", "password")

    # Allow multiple users (format: "user1:pass1,user2:pass2,...")
    USERS: dict = {}

    @classmethod
    def load_users(cls):
        """Load users from environment."""
        if cls.USERS:
            return

        # First, try to load from AUTH_USERS env var
        users_env = os.getenv("AUTH_USERS", "")
        if users_env:
            try:
                for user_pair in users_env.split(","):
                    if ":" in user_pair:
                        username, password = user_pair.strip().split(":", 1)
                        cls.USERS[username.strip()] = password.strip()
                logger.info(f"Loaded {len(cls.USERS)} users from AUTH_USERS")
            except Exception as e:
                logger.warning(f"Failed to parse AUTH_USERS: {e}")

        # If no users loaded, use defaults
        if not cls.USERS:
            cls.USERS[cls.DEFAULT_USERNAME] = cls.DEFAULT_PASSWORD
            logger.info("Using default credentials")


class AuthManager:
    """Manages authentication and session."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against hash."""
        return AuthManager.hash_password(password) == stored_hash

    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is currently authenticated."""
        return st.session_state.get("authenticated", False)

    @staticmethod
    def get_current_user() -> Optional[str]:
        """Get currently logged-in username."""
        return st.session_state.get("username", None)

    @staticmethod
    def login(username: str, password: str) -> Tuple[bool, str]:
        """Attempt login with username and password.

        Returns:
            Tuple of (success, message)
        """
        AuthConfig.load_users()

        # Check if user exists
        if username not in AuthConfig.USERS:
            logger.warning(f"Login failed: user '{username}' not found")
            return False, "Invalid username or password"

        # Check password
        stored_password = AuthConfig.USERS[username]
        if password != stored_password:
            logger.warning(f"Login failed: invalid password for '{username}'")
            return False, "Invalid username or password"

        # Login successful
        logger.info(f"User logged in: {username}")
        st.session_state.authenticated = True
        st.session_state.username = username
        return True, f"Welcome, {username}!"

    @staticmethod
    def logout():
        """Logout current user."""
        username = AuthManager.get_current_user()
        st.session_state.authenticated = False
        st.session_state.username = None
        if username:
            logger.info(f"User logged out: {username}")


def show_login_form() -> bool:
    """Display login form. Returns True if login successful."""
    st.title("🔐 Design Analysis Studio - Login")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Please sign in to continue")

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )

        col_login, col_demo = st.columns(2)

        with col_login:
            if st.button("Sign In", type="primary", use_container_width=True):
                if username and password:
                    success, message = AuthManager.login(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter both username and password")

        with col_demo:
            if st.button("Demo Credentials", use_container_width=True):
                st.info(
                    f"Username: `{AuthConfig.DEFAULT_USERNAME}`\n\n"
                    f"Password: `{AuthConfig.DEFAULT_PASSWORD}`"
                )

        st.divider()
        st.markdown(
            """
            **About Authentication**
            
            This app uses optional authentication. Set `AUTH_ENABLED=true` environment variable to enable.
            
            Credentials can be configured via:
            - `AUTH_USERNAME` / `AUTH_PASSWORD` (single user)
            - `AUTH_USERS` (multiple users: "user1:pass1,user2:pass2")
            
            For development, authentication is disabled by default.
            """
        )

    return False


def show_auth_status():
    """Show authentication status in sidebar."""
    if not AuthConfig.AUTH_ENABLED:
        return

    current_user = AuthManager.get_current_user()
    if current_user:
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            st.sidebar.markdown(f"**👤 {current_user}**")
        with col2:
            if st.sidebar.button("🚪 Logout", key="logout_btn"):
                AuthManager.logout()
                st.rerun()
    else:
        st.sidebar.warning("Not authenticated")


def require_auth() -> bool:
    """
    Wrapper function to enforce authentication.
    Returns True if user is authenticated or auth is disabled.
    """
    if not AuthConfig.AUTH_ENABLED:
        # Auth disabled - allow access
        return True

    if AuthManager.is_authenticated():
        # User is authenticated
        return True

    # Auth required but not authenticated
    show_login_form()
    st.stop()


def get_user_info() -> dict:
    """Get current user info."""
    return {
        "authenticated": AuthManager.is_authenticated(),
        "username": AuthManager.get_current_user(),
        "auth_enabled": AuthConfig.AUTH_ENABLED,
    }
