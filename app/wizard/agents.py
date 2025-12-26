"""Agent Registry and Management for AIpowered365 Labs"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Agent:
    """Represents an AI analysis agent."""
    id: str
    name: str
    icon: str
    description: str
    default_enabled: bool = False


def get_available_agents() -> List[Agent]:
    """
    Get list of available analysis agents.

    Returns:
        List of Agent objects with metadata
    """
    return [
        Agent(
            id="visual_analysis",
            name="Visual Analyst",
            icon="🎨",
            description="Analyzes composition, colors, typography, and visual hierarchy",
            default_enabled=True
        ),
        Agent(
            id="ux_critique",
            name="UX Critique",
            icon="👥",
            description="Evaluates usability, user flow, and interaction patterns",
            default_enabled=True
        ),
        Agent(
            id="market_fit_analysis",
            name="Market Fit",
            icon="📊",
            description="Assesses target audience appeal and market positioning",
            default_enabled=False
        ),
        Agent(
            id="conversion_optimization",
            name="Conversion",
            icon="💰",
            description="Identifies optimization opportunities for conversions and engagement",
            default_enabled=False
        ),
        Agent(
            id="brand_alignment",
            name="Brand Alignment",
            icon="🏢",
            description="Checks consistency with brand guidelines and identity",
            default_enabled=False
        ),
    ]


def get_agent_by_id(agent_id: str) -> Optional[Agent]:
    """
    Get a single agent by ID.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent object or None if not found
    """
    agents = get_available_agents()
    for agent in agents:
        if agent.id == agent_id:
            return agent
    return None


def get_preset_agents(preset: str) -> List[str]:
    """
    Get agent IDs for a given preset.

    Args:
        preset: One of "quick", "full", or "custom"

    Returns:
        List of agent IDs
    """
    presets = {
        "quick": ["visual_analysis", "ux_critique"],
        "full": ["visual_analysis", "ux_critique", "market_fit_analysis",
                 "conversion_optimization", "brand_alignment"],
        "custom": []
    }
    return presets.get(preset, [])


def validate_agent_selection(agent_ids: List[str]) -> tuple[bool, str]:
    """
    Validate a list of selected agent IDs.

    Args:
        agent_ids: List of agent IDs to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not agent_ids:
        return False, "At least one agent must be selected"

    available_ids = [a.id for a in get_available_agents()]
    for agent_id in agent_ids:
        if agent_id not in available_ids:
            return False, f"Unknown agent: {agent_id}"

    return True, ""
