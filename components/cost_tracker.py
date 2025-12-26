"""
Cost tracking for API calls and token usage.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from components.models import APICallMetrics, TokenUsage


@dataclass
class CostTracker:
    """Track API costs and token usage per session."""

    session_id: str
    api_calls: List[APICallMetrics] = field(default_factory=list)

    # Pricing per 1K tokens (OpenRouter pricing as of 2024)
    PRICING: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "openai/gpt-4o": {"input": 0.0025, "output": 0.010},
            "openai/text-embedding-3-small": {"input": 0.00002, "output": 0.0},
            "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
            "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "google/gemini-pro-vision": {"input": 0.001, "output": 0.002},
        }
    )

    def record_api_call(
        self,
        model: str,
        agent: str,
        latency_ms: float,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> APICallMetrics:
        """
        Record an API call with usage and cost.

        Args:
            model: Model name (e.g., "openai/gpt-4o")
            agent: Agent name (e.g., "visual")
            latency_ms: API call latency in milliseconds
            tokens_in: Input tokens (optional)
            tokens_out: Output tokens (optional)
            cost_usd: Explicit cost (optional; auto-calculated if not provided)

        Returns:
            APICallMetrics instance
        """
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = self._calculate_cost(model, tokens_in, tokens_out)

        # Create token usage if provided
        token_usage = None
        if tokens_in is not None or tokens_out is not None:
            token_usage = TokenUsage(
                input_tokens=tokens_in or 0,
                output_tokens=tokens_out or 0,
                total_tokens=(tokens_in or 0) + (tokens_out or 0),
            )

        # Create metrics
        metrics = APICallMetrics(
            model=model,
            agent=agent,
            latency_ms=latency_ms,
            tokens=token_usage,
            cost_usd=cost_usd,
            timestamp=datetime.utcnow(),
        )

        self.api_calls.append(metrics)
        return metrics

    def _calculate_cost(
        self,
        model: str,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
    ) -> float:
        """
        Calculate cost for an API call.

        Args:
            model: Model name
            tokens_in: Input tokens
            tokens_out: Output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in self.PRICING:
            # Unknown model - return 0
            return 0.0

        pricing = self.PRICING[model]
        cost = 0.0

        if tokens_in and tokens_in > 0:
            cost += (tokens_in / 1000.0) * pricing.get("input", 0.0)

        if tokens_out and tokens_out > 0:
            cost += (tokens_out / 1000.0) * pricing.get("output", 0.0)

        return cost

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        return sum(
            call.tokens.total_tokens for call in self.api_calls if call.tokens
        )

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD across all calls."""
        return sum(call.cost_usd for call in self.api_calls)

    @property
    def total_latency_ms(self) -> float:
        """Total latency in milliseconds."""
        return sum(call.latency_ms for call in self.api_calls)

    def to_dict(self) -> Dict:
        """Export tracker data as dictionary."""
        return {
            "session_id": self.session_id,
            "total_calls": len(self.api_calls),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "api_calls": [
                {
                    "model": call.model,
                    "agent": call.agent,
                    "latency_ms": call.latency_ms,
                    "tokens": (
                        {
                            "input": call.tokens.input_tokens,
                            "output": call.tokens.output_tokens,
                            "total": call.tokens.total_tokens,
                        }
                        if call.tokens
                        else None
                    ),
                    "cost_usd": round(call.cost_usd, 6),
                    "timestamp": call.timestamp.isoformat(),
                }
                for call in self.api_calls
            ],
        }

    def to_json(self) -> str:
        """Export tracker data as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"📊 Cost Summary (Session: {self.session_id})",
            f"  Calls: {len(self.api_calls)}",
            f"  Tokens: {self.total_tokens:,}",
            f"  Cost: ${self.total_cost_usd:.6f}",
            f"  Latency: {self.total_latency_ms:.0f}ms",
        ]
        return "\n".join(lines)
