# """
# GymProductionServer Framework

# This framework provides a base class for creating MCP servers that wrap gymnasium
# environments using adapters. It handles:

# 1. Multi-session and single-session server lifecycle
# 2. Automatic tool and resource registration
# 3. Environment management via adapters
# 4. MCP resource patterns for initial state
# 5. Standardized tool signatures
# 6. Session management with proper seed extraction

# Usage:
#     class MyGameProdServer(GymProductionServer):
#         def __init__(self):
#             super().__init__("MyGame-v1", MyAdapter())

#         def _register_tools(self):
#             # Register domain-specific tools

#         @staticmethod
#         def format_observation(obs, env):
#             # Format observations for MCP responses
# """

# from abc import ABC, abstractmethod
# from typing import Any, Dict

# from .adapter import EnvironmentAdapter


# class GymProductionServer(ABC):
#     """
#     Multi-session capable MCP server base class.

#     Subclasses supply:
#     • adapter - EnvironmentAdapter instance
#     • _register_tools() - add ergonomic tools
#     • format_observation(obs, env) - env-specific view dict
#     """

#     def __init__(self, name: str, adapter: EnvironmentAdapter):
#         """
#         Initialize production server.

#         Args:
#             name: Server name for MCP
#             adapter: Environment adapter instance
#         """
#         # Core infrastructure moved to McpGym
#         # This base class now only provides the abstract interface
#         pass





#     # Abstract methods that subclasses must implement

#     @abstractmethod
#     def _register_tools(self):
#         """Register domain-specific MCP tools."""
#         pass

#     @staticmethod
#     @abstractmethod
#     def format_observation(obs: Any, env: Any) -> Dict[str, Any]:
#         """Format observation for MCP response."""
#         pass


