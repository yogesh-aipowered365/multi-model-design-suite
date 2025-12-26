"""
Integration layer between orchestration, agents, and caching system.
Provides high-level functions for cached agent execution.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from components.caching import get_cache_manager
from components.models import AgentResult, AnalysisMode
from components.guardrails import Guardrails

logger = logging.getLogger(__name__)


class CachedAgentExecutor:
    """
    Executes agents with automatic caching based on inputs.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize executor.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.cache_manager = get_cache_manager() if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

    def execute_with_cache(
        self,
        agent_func,
        agent_name: str,
        image_hash: str,
        platform: str,
        creative_type: str,
        agent_version: str = "1.0",
        **agent_kwargs,
    ) -> AgentResult:
        """
        Execute agent with caching.

        Args:
            agent_func: Function to execute
            agent_name: Name of agent
            image_hash: SHA256 hash of image
            platform: Platform name
            creative_type: Creative type
            agent_version: Agent version (for cache invalidation)
            **agent_kwargs: Arguments to pass to agent function

        Returns:
            AgentResult with cache metadata
        """
        # Try to get from cache
        if self.use_cache:
            cached_result = self.cache_manager.get_agent_result(
                image_hash=image_hash,
                agent_name=agent_name,
                platform=platform,
                creative_type=creative_type,
                agent_version=agent_version,
            )

            if cached_result:
                # Return cached result with metadata
                result = AgentResult(**cached_result)
                result.was_cached = True
                result.cache_age_seconds = time.time()  # Will be updated by caller

                self.cache_hits += 1
                logger.debug(f"Cache HIT: {agent_name} ({image_hash[:8]}...)")
                return result

            self.cache_misses += 1

        # Execute agent
        logger.debug(
            f"Cache MISS: {agent_name} ({image_hash[:8]}...) - executing")
        start_time = time.time()
        try:
            result = agent_func(**agent_kwargs)

            # Convert to dict for caching
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result

            # Cache result
            if self.use_cache:
                self.cache_manager.cache_agent_result(
                    image_hash=image_hash,
                    agent_name=agent_name,
                    platform=platform,
                    creative_type=creative_type,
                    result=result_dict,
                    agent_version=agent_version,
                )

            # Ensure we return AgentResult
            if not isinstance(result, AgentResult):
                result = AgentResult(**result_dict)

            result.was_cached = False
            return result

        except Exception as e:
            logger.error(f"Agent execution failed: {agent_name} - {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for this execution.

        Returns:
            Dict with cache stats
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total if total > 0 else 0
        )

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_calls": total,
            "cache_hit_rate": round(hit_rate, 3),
        }


class TokenEstimator:
    """
    Estimates and tracks token usage across analysis.
    """

    def __init__(self):
        """Initialize estimator."""
        self.estimates = {}
        self.actual_usage = {}

    def estimate_analysis(
        self,
        num_images: int,
        analysis_mode: AnalysisMode = AnalysisMode.STANDARD,
        is_comparison: bool = False,
        enabled_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate token usage for analysis.

        Args:
            num_images: Number of images
            analysis_mode: Analysis mode
            is_comparison: Whether in comparison mode
            enabled_agents: Specific agents (if None, use mode default)

        Returns:
            Token estimate dict
        """
        estimate = Guardrails.estimate_token_usage(
            num_images=num_images,
            analysis_mode=analysis_mode,
            is_comparison=is_comparison,
            enabled_agents=enabled_agents,
        )

        self.estimates = estimate
        return estimate

    def add_actual_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Add actual token usage from an agent.

        Args:
            agent_name: Agent name
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        total = input_tokens + output_tokens
        self.actual_usage[agent_name] = {
            "input": input_tokens,
            "output": output_tokens,
            "total": total,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get token usage summary.

        Returns:
            Summary dict
        """
        total_actual = sum(u["total"] for u in self.actual_usage.values())

        return {
            "estimated": self.estimates.get("total_tokens", 0),
            "actual": total_actual,
            "estimate_accuracy": round(
                (1 - abs(self.estimates.get("total_tokens", total_actual) - total_actual) /
                 max(self.estimates.get("total_tokens", 1), total_actual)),
                3
            ) if self.estimates else None,
            "by_agent": self.actual_usage,
        }


class AnalysisOrchestrator:
    """
    High-level orchestration with caching and guardrails.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize orchestrator.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.executor = CachedAgentExecutor(use_cache=use_cache)
        self.token_estimator = TokenEstimator()
        self.execution_log = []

    def validate_and_estimate(
        self,
        images: List[tuple],  # (filename, bytes)
        analysis_mode: str = "standard",
        is_comparison: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate images and provide cost/token estimate.

        Args:
            images: List of (filename, bytes) tuples
            analysis_mode: 'fast', 'standard', or 'comprehensive'
            is_comparison: Whether in comparison mode

        Returns:
            Dict with validation result and estimate
        """
        # Validate images
        is_valid, errors, warnings = Guardrails.validate_images(images)

        # Get token estimate
        try:
            mode = AnalysisMode(analysis_mode)
        except ValueError:
            mode = AnalysisMode.STANDARD

        estimate = self.token_estimator.estimate_analysis(
            num_images=len(images),
            analysis_mode=mode,
            is_comparison=is_comparison,
        )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "estimate": estimate,
            "formatted_estimate": {
                "tokens": Guardrails.format_token_display(estimate["total_tokens"]),
                "cost": Guardrails.format_cost_display(
                    estimate["estimated_cost"],
                    len(images),
                ),
                "time_seconds": estimate["processing_time_seconds"],
            },
        }

    def log_execution(self, event: str, details: Dict[str, Any] = None) -> None:
        """
        Log execution event.

        Args:
            event: Event description
            details: Additional details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "details": details or {},
        }
        self.execution_log.append(entry)
        logger.debug(f"Orchestration: {event}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.

        Returns:
            Summary dict
        """
        return {
            "cache_stats": self.executor.get_stats(),
            "token_summary": self.token_estimator.get_summary(),
            "execution_log": self.execution_log,
        }
