"""
Reward-Kit MCP Integration Framework

This module provides utilities for creating MCP servers that integrate
with reward-kit environments and evaluation workflows.

It also provides the refactored MCP environment components for better modularity.
"""

from .adapter import EnvironmentAdapter

# New refactored components
from .client import MCPConnectionManager
from .execution import FireworksPolicy, LLMBasePolicy, ExecutionManager

# North Star MCP-Gym Framework
from .mcpgym import McpGym
from .server import MCPEnvironmentServer
from .session import GeneralMCPVectorEnv
from .simulation_server import SimulationServerBase
from .types import DatasetRow, MCPSession, MCPToolCall, Trajectory

__all__ = [
    # Legacy MCP server components
    "MCPEnvironmentServer",
    "EnvironmentAdapter",
    "SimulationServerBase",
    # New refactored components
    "MCPConnectionManager",
    "LLMBasePolicy",
    "FireworksPolicy",
    "ExecutionManager",
    "GeneralMCPVectorEnv",
    "MCPSession",
    "MCPToolCall",
    "DatasetRow",
    "Trajectory",
    # North Star MCP-Gym Framework
    "McpGym",
]
