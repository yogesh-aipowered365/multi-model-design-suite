# components/llm_client.py

"""
LLM Client Wrapper - FIXED VERSION
Unified interface for vision and text LLM calls with retry logic, timeouts, and usage tracking

FIX: Properly format images for Claude models via OpenRouter
"""

import os
import json
import time
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8501")
APP_NAME = os.getenv("APP_NAME", "DesignAnalysisPoc")

# Model selection from environment
VISION_MODEL = os.getenv("LLM_VISION_MODEL", "openai/gpt-4o")
TEXT_MODEL = os.getenv("LLM_TEXT_MODEL", "openai/gpt-4-turbo")

# Retry and timeout settings
DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT = 60
DEFAULT_BACKOFF = 1.5

# Try to import OpenAI client if available
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        USE_OPENAI = False
        openai_client = None
else:
    openai_client = None


@dataclass
class LLMResponse:
    """Unified LLM response format"""
    text: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    model: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "error": self.error
        }


def run_llm(
    task_name: str,
    messages: List[Dict[str, str]],
    images: Optional[List[str]] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    json_mode: bool = False,
    api_key: Optional[str] = None
) -> LLMResponse:
    """
    Unified LLM interface supporting vision and text calls

    Args:
        task_name: Name of the task (for logging)
        messages: List of message dicts with 'role' and 'content' keys
        images: Optional list of base64 encoded images (for vision tasks)
        model: Override default model selection
        temperature: Generation temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        retries: Number of retry attempts
        timeout: Request timeout in seconds
        json_mode: Request JSON-formatted response
        api_key: Optional API key (BYOK: Bring Your Own Key). If provided, overrides env var

    Returns:
        LLMResponse: {text, usage, latency_ms, model, error}

    Examples:
        # Text-only call
        response = run_llm(
            "analyze",
            messages=[{"role": "user", "content": "What is design?"}]
        )

        # Vision call with user API key
        response = run_llm(
            "visual_analysis",
            messages=[{"role": "user", "content": "Analyze this design"}],
            images=["data:image/jpeg;base64,..."],
            api_key="sk-or-v1-..."
        )
    """

    start_time = time.time()
    selected_model = model or (VISION_MODEL if images else TEXT_MODEL)

    # BYOK: Check if API key is provided or in Streamlit session
    effective_api_key = api_key

    # Try to get API key from Streamlit session if not provided
    if not effective_api_key:
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and 'api_key' in st.session_state:
                effective_api_key = st.session_state.api_key
        except ImportError:
            pass

    # Fall back to environment variable
    if not effective_api_key:
        effective_api_key = OPENROUTER_API_KEY

    # Use OpenAI if available and keys match
    if USE_OPENAI and OPENAI_API_KEY:
        return _run_openai_llm(
            task_name, messages, images, selected_model,
            temperature, max_tokens, retries, timeout, json_mode, start_time
        )
    elif effective_api_key:
        return _run_openrouter_llm(
            task_name, messages, images, selected_model,
            temperature, max_tokens, retries, timeout, json_mode, start_time,
            api_key=effective_api_key
        )
    else:
        error_msg = "No LLM API key configured (OPENAI_API_KEY or OPENROUTER_API_KEY). Please provide API key."
        print(f"❌ {error_msg}")
        latency_ms = (time.time() - start_time) * 1000
        return LLMResponse(
            text="",
            usage=None,
            latency_ms=latency_ms,
            model=selected_model,
            error=error_msg
        )


def _run_openai_llm(
    task_name: str,
    messages: List[Dict[str, str]],
    images: Optional[List[str]],
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: int,
    json_mode: bool,
    start_time: float
) -> LLMResponse:
    """Internal: Run LLM call via OpenAI API"""

    # Build content for messages with optional images
    content = []

    # Add text from first message
    if messages and messages[0].get("content"):
        content.append({
            "type": "text",
            "text": messages[0]["content"]
        })

    # Add images if provided (vision task)
    if images:
        for image_url in images:
            if isinstance(image_url, str):
                # Ensure proper data URL format
                if image_url.startswith("data:"):
                    url = image_url
                else:
                    url = f"data:image/jpeg;base64,{image_url}"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

    # Attempt call with retries
    last_error = None
    for attempt in range(retries + 1):
        try:
            # Determine actual model for OpenAI (map openai/* models)
            actual_model = model.replace(
                "openai/", "") if "/" in model else model

            # Select vision or text model appropriately
            if images:
                actual_model = "gpt-4-turbo" if "vision" not in actual_model else actual_model

            kwargs = {
                "model": actual_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = openai_client.chat.completions.create(**kwargs)

            # Extract response
            text = response.choices[0].message.content

            # Extract usage if available
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            latency_ms = (time.time() - start_time) * 1000

            print(
                f"✅ {task_name} completed via OpenAI ({actual_model}) in {latency_ms:.0f}ms")

            return LLMResponse(
                text=text,
                usage=usage,
                latency_ms=latency_ms,
                model=actual_model,
                error=None
            )

        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                sleep_time = DEFAULT_BACKOFF ** attempt
                print(
                    f"⚠️ {task_name} OpenAI error: {e}. Retry {attempt + 1}/{retries} in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(
                    f"❌ {task_name} OpenAI failed after {retries + 1} attempts: {e}")

    latency_ms = (time.time() - start_time) * 1000
    return LLMResponse(
        text="",
        usage=None,
        latency_ms=latency_ms,
        model=model,
        error=last_error or "Unknown error"
    )


def _run_openrouter_llm(
    task_name: str,
    messages: List[Dict[str, str]],
    images: Optional[List[str]],
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: int,
    json_mode: bool,
    start_time: float,
    api_key: Optional[str] = None
) -> LLMResponse:
    """Internal: Run LLM call via OpenRouter API - FIXED for Claude models"""

    # Use provided API key or fall back to environment variable
    effective_api_key = api_key or OPENROUTER_API_KEY

    if not effective_api_key:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = "No OpenRouter API key provided (BYOK). Please provide your API key."
        return LLMResponse(
            text="",
            usage=None,
            latency_ms=latency_ms,
            model=model,
            error=error_msg
        )

    # Build content for messages with optional images
    content = []

    # Add text from first message
    if messages and messages[0].get("content"):
        content.append({
            "type": "text",
            "text": messages[0]["content"]
        })

    # Add images if provided (vision task)
    # FIX: Check if this is a Claude model and use proper format
    if images:
        is_claude_model = "claude" in model.lower()
        
        for image_data in images:
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if image_data.startswith("data:image/"):
                    # Extract base64 part only
                    base64_str = image_data.split("base64,", 1)[-1]
                else:
                    base64_str = image_data

                # FIXED: Use correct format for Claude models
                if is_claude_model:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_str
                        }
                    })
                else:
                    # For OpenAI models via OpenRouter, use image_url format
                    url = f"data:image/jpeg;base64,{base64_str}"
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })

    # Build request headers
    headers = {
        "Authorization": f"Bearer {effective_api_key}",
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }

    # Build request payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Attempt call with retries
    last_error = None
    for attempt in range(retries + 1):
        try:
            base_url = OPENROUTER_BASE_URL.rstrip("/")
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )

            # Check response status
            if response.status_code != 200:
                try:
                    error_detail = response.json()
                except ValueError:
                    error_detail = response.text

                error_msg = f"HTTP {response.status_code}: {error_detail}"
                raise Exception(error_msg)

            # Parse response
            result = response.json()
            choices = result.get("choices", [])

            if not choices or not choices[0].get("message"):
                raise Exception("Empty response from model")

            text = choices[0]["message"].get("content", "")

            if not text:
                raise Exception("Empty content in response")

            # Extract usage if available
            usage = None
            if result.get("usage"):
                usage = {
                    "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                    "completion_tokens": result["usage"].get("completion_tokens", 0),
                    "total_tokens": result["usage"].get("total_tokens", 0)
                }

            latency_ms = (time.time() - start_time) * 1000

            print(
                f"✅ {task_name} completed via OpenRouter ({model}) in {latency_ms:.0f}ms")

            return LLMResponse(
                text=text,
                usage=usage,
                latency_ms=latency_ms,
                model=model,
                error=None
            )

        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                sleep_time = DEFAULT_BACKOFF ** attempt
                print(
                    f"⚠️ {task_name} OpenRouter error: {e}. Retry {attempt + 1}/{retries} in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(
                    f"❌ {task_name} OpenRouter failed after {retries + 1} attempts: {e}")

    latency_ms = (time.time() - start_time) * 1000
    return LLMResponse(
        text="",
        usage=None,
        latency_ms=latency_ms,
        model=model,
        error=last_error or "Unknown error"
    )


def parse_json_response(response: LLMResponse) -> Dict[str, Any]:
    """
    Parse JSON from LLM response text

    Args:
        response: LLMResponse object

    Returns:
        dict: Parsed JSON or error dict
    """

    if response.error:
        return {"error": response.error}

    if not response.text:
        return {"error": "Empty response"}

    try:
        parsed = json.loads(response.text)

        # Handle nested JSON strings
        if isinstance(parsed, dict):
            for key, value in list(parsed.items()):
                if isinstance(value, str) and value.strip().startswith("{"):
                    try:
                        parsed[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass

        return parsed

    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON response: {e}",
            "raw_text": response.text[:500]  # First 500 chars
        }
