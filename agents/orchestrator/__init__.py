"""
Agent Orchestrator Module for AegisMedBot Hospital Intelligence Platform

This module serves as the central nervous system of the multi-agent AI platform.
It coordinates all specialized agents, manages task delegation, maintains conversation
context, and ensures proper communication flow between different components.

The orchestrator implements a hierarchical agent architecture where a root orchestrator
agent routes user requests to specialized sub-agents based on intent analysis,
domain expertise requirements, and confidence scoring.

Author: AegisMedBot Team
Version: 1.0.0
"""

# Import core orchestrator components to expose them at package level
# This allows other modules to import directly from agents.orchestrator
from .agent_orchestrator import AgentOrchestrator
from .task_delegator import TaskDelegator
from .context_manager import ContextManager

# Define what gets imported when using "from agents.orchestrator import *"
# This controls the public API of the orchestrator package
__all__ = [
    'AgentOrchestrator',    # Main orchestrator class that coordinates all agents
    'TaskDelegator',        # Handles intelligent routing of tasks to appropriate agents
    'ContextManager'        # Manages conversation context across multiple turns
]

# Package metadata for documentation and debugging
__version__ = '1.0.0'
__author__ = 'AegisMedBot Team'
__description__ = 'Multi-agent orchestration layer for hospital intelligence platform'