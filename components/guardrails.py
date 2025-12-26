"""
Guardrails and validation layer.
Enforces constraints on inputs, shows token/cost estimates, and enables fast mode.
"""

from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import io
import base64
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AnalysisMode(str, Enum):
    """Analysis modes with different token/cost profiles."""
    FAST = "fast"  # Minimal agents, ~5K tokens, ~$0.02
    STANDARD = "standard"  # All agents, ~25K tokens, ~$0.10
    COMPREHENSIVE = "comprehensive"  # All agents + deep analysis, ~50K tokens, ~$0.20


class Guardrails:
    """
    Enforces constraints and provides token/cost estimates.

    Constraints:
    - Max 5 images
    - Image resolution downscaling
    - Token/cost estimation
    - Fast mode support
    """

    # Configuration constants
    MAX_IMAGES = 5
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048
    MIN_WIDTH = 256
    MIN_HEIGHT = 256
    MAX_FILE_SIZE_MB = 50

    # Agent token costs (estimated tokens per image analysis)
    AGENT_TOKENS = {
        "visual_analysis": 3000,  # CLIP embedding + visual description
        "ux_critique": 4000,  # Detailed UX analysis
        "market_research": 3500,  # Market positioning
        "conversion_cta": 3500,  # CTA analysis
        "brand_consistency": 3000,  # Brand analysis
        "design_comparison": 6000,  # Comparison analysis (per design pair)
    }

    # Fast mode uses subset of agents
    FAST_MODE_AGENTS = ["market_research", "conversion_cta"]
    STANDARD_MODE_AGENTS = ["visual_analysis", "ux_critique",
                            "market_research", "conversion_cta", "brand_consistency"]

    # Base overhead tokens (routing, parsing, formatting)
    BASE_TOKENS = {
        AnalysisMode.FAST: 2000,
        AnalysisMode.STANDARD: 3000,
        AnalysisMode.COMPREHENSIVE: 5000,
    }

    # Token to cost ratio (assuming Claude 3.5 Sonnet pricing)
    # Input: $0.003 / 1M tokens
    # Output: $0.015 / 1M tokens
    # Average: ~$0.009 / 1K tokens
    COST_PER_1K_TOKENS = 0.009

    @staticmethod
    def validate_images(
        images: List[Tuple[str, bytes]],
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate images against guardrails.

        Args:
            images: List of (filename, bytes) tuples

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check max images
        if len(images) > Guardrails.MAX_IMAGES:
            errors.append(
                f"Maximum {Guardrails.MAX_IMAGES} images allowed, got {len(images)}"
            )

        for filename, file_bytes in images:
            # Check file size
            size_mb = len(file_bytes) / (1024 * 1024)
            if size_mb > Guardrails.MAX_FILE_SIZE_MB:
                errors.append(
                    f"{filename}: File too large ({size_mb:.1f}MB > {Guardrails.MAX_FILE_SIZE_MB}MB)"
                )
                continue

            # Try to load image
            try:
                image = Image.open(io.BytesIO(file_bytes))
                width, height = image.size

                # Check resolution
                if width < Guardrails.MIN_WIDTH or height < Guardrails.MIN_HEIGHT:
                    errors.append(
                        f"{filename}: Image too small ({width}x{height} < {Guardrails.MIN_WIDTH}x{Guardrails.MIN_HEIGHT})"
                    )

                if width > Guardrails.MAX_WIDTH or height > Guardrails.MAX_HEIGHT:
                    warnings.append(
                        f"{filename}: Image resolution will be downscaled ({width}x{height} > {Guardrails.MAX_WIDTH}x{Guardrails.MAX_HEIGHT})"
                    )

            except Exception as e:
                errors.append(f"{filename}: Invalid image file ({str(e)})")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    @staticmethod
    def downscale_image_if_needed(
        image: Image.Image,
        max_width: int = MAX_WIDTH,
        max_height: int = MAX_HEIGHT,
    ) -> Image.Image:
        """
        Downscale image if it exceeds max dimensions.

        Args:
            image: PIL Image
            max_width: Maximum width
            max_height: Maximum height

        Returns:
            Downscaled image (or original if smaller)
        """
        width, height = image.size

        if width <= max_width and height <= max_height:
            return image

        # Calculate scaling factor to fit within bounds
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.info(
            f"Downscaling image from {width}x{height} to {new_width}x{new_height}"
        )
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def estimate_token_usage(
        num_images: int,
        analysis_mode: AnalysisMode = AnalysisMode.STANDARD,
        is_comparison: bool = False,
        enabled_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate token usage and cost for analysis.

        Args:
            num_images: Number of images to analyze
            analysis_mode: Analysis mode (fast/standard/comprehensive)
            is_comparison: Whether this is comparison mode
            enabled_agents: Specific agents to enable (if None, use mode default)

        Returns:
            Dict with token/cost estimates
        """
        # Determine which agents to use
        if enabled_agents:
            agents = enabled_agents
        elif analysis_mode == AnalysisMode.FAST:
            agents = Guardrails.FAST_MODE_AGENTS
        else:
            agents = Guardrails.STANDARD_MODE_AGENTS

        # Calculate base tokens
        base = Guardrails.BASE_TOKENS[analysis_mode]

        # Calculate agent tokens per image
        agent_tokens = sum(
            Guardrails.AGENT_TOKENS.get(agent, 3000) for agent in agents
        )

        # Total tokens for all images
        image_tokens = agent_tokens * num_images

        # Add comparison overhead if needed
        comparison_tokens = 0
        if is_comparison and num_images > 1:
            # Comparison between pairs
            num_pairs = num_images * (num_images - 1) // 2
            comparison_tokens = Guardrails.AGENT_TOKENS.get(
                "design_comparison", 6000) * num_pairs

        total_tokens = base + image_tokens + comparison_tokens

        # Calculate cost (rough estimate)
        estimated_cost = total_tokens * Guardrails.COST_PER_1K_TOKENS / 1000

        return {
            "num_images": num_images,
            "num_agents": len(agents),
            "agents": agents,
            "analysis_mode": analysis_mode.value,
            "is_comparison": is_comparison,
            "base_tokens": base,
            "image_tokens_per_agent": agent_tokens // len(agents) if agents else 0,
            "total_image_tokens": image_tokens,
            "comparison_tokens": comparison_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": round(estimated_cost, 3),
            "estimated_cost_range": {
                "low": round(estimated_cost * 0.8, 3),  # 20% variance
                "high": round(estimated_cost * 1.2, 3),
            },
            "processing_time_seconds": {
                "fast": 30,
                "standard": 60,
                "comprehensive": 120,
            }[analysis_mode.value],
        }

    @staticmethod
    def get_fast_mode_summary() -> Dict[str, Any]:
        """
        Get summary of fast mode characteristics.

        Returns:
            Dict with fast mode info
        """
        return {
            "name": "Fast Mode",
            "description": "Quick analysis with essential agents",
            "agents": Guardrails.FAST_MODE_AGENTS,
            "estimated_cost": Guardrails.estimate_token_usage(
                num_images=1,
                analysis_mode=AnalysisMode.FAST,
            )["estimated_cost"],
            "estimated_time_seconds": 30,
            "benefits": [
                "✓ Lower cost (~$0.02 per image)",
                "✓ Faster analysis (~30 seconds)",
                "✓ Focuses on market & conversion metrics",
            ],
            "tradeoffs": [
                "⚠ Skips visual & UX analysis",
                "⚠ Less detailed findings",
                "⚠ Reduced recommendations",
            ],
        }

    @staticmethod
    def get_standard_mode_summary() -> Dict[str, Any]:
        """
        Get summary of standard mode characteristics.

        Returns:
            Dict with standard mode info
        """
        return {
            "name": "Standard Mode",
            "description": "Comprehensive analysis with all agents",
            "agents": Guardrails.STANDARD_MODE_AGENTS,
            "estimated_cost": Guardrails.estimate_token_usage(
                num_images=1,
                analysis_mode=AnalysisMode.STANDARD,
            )["estimated_cost"],
            "estimated_time_seconds": 60,
            "benefits": [
                "✓ Full agent coverage",
                "✓ Detailed visual, UX, market, conversion, brand analysis",
                "✓ Comprehensive recommendations",
            ],
            "tradeoffs": [
                "⚠ Higher cost (~$0.10 per image)",
                "⚠ Slower analysis (~60 seconds)",
            ],
        }

    @staticmethod
    def format_cost_display(total_cost: float, num_images: int) -> str:
        """
        Format cost for display.

        Args:
            total_cost: Total estimated cost
            num_images: Number of images

        Returns:
            Formatted string
        """
        per_image = total_cost / num_images if num_images > 0 else 0
        return f"${total_cost:.3f} total (${per_image:.3f}/image)"

    @staticmethod
    def format_token_display(total_tokens: int) -> str:
        """
        Format tokens for display.

        Args:
            total_tokens: Total tokens

        Returns:
            Formatted string
        """
        if total_tokens >= 1_000_000:
            return f"{total_tokens / 1_000_000:.1f}M tokens"
        elif total_tokens >= 1_000:
            return f"{total_tokens / 1_000:.1f}K tokens"
        else:
            return f"{total_tokens} tokens"


# Validation helper functions
def validate_and_downscale_images(
    images_dict: Dict[str, bytes],
) -> Tuple[Dict[str, Image.Image], List[str]]:
    """
    Validate and downscale images.

    Args:
        images_dict: Dict of filename -> file bytes

    Returns:
        Tuple of (processed_images_dict, warnings)
    """
    # Validate
    is_valid, errors, warnings = Guardrails.validate_images(
        list(images_dict.items())
    )

    if not is_valid:
        raise ValueError(f"Image validation failed: {'; '.join(errors)}")

    # Downscale
    processed = {}
    for filename, file_bytes in images_dict.items():
        image = Image.open(io.BytesIO(file_bytes))
        image = Guardrails.downscale_image_if_needed(image)
        processed[filename] = image

    return processed, warnings
