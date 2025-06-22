"""
FrozenLake MCP Server - Production Implementation

This is a pure, standard MCP server that uses the reward-kit
framework for environment integration. This demonstrates the
clean separation of concerns:

1. MCP Server (this file) - Pure MCP protocol handling
2. Framework Code (reward_kit.mcp) - Generic MCP utilities
3. User Code (frozen_lake_adapter.py) - Environment-specific logic
"""

import argparse

from frozen_lake_adapter import FrozenLakeAdapter

from reward_kit.mcp import MCPEnvironmentServer


def main():
    """Main entry point for FrozenLake MCP server."""
    parser = argparse.ArgumentParser(description="FrozenLake MCP Server")
    parser.add_argument(
        "--transport",
        default="sse",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--map-name", default="4x4", choices=["4x4", "8x8"], help="FrozenLake map size"
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        default=True,
        help="Enable slippery ice (stochastic transitions)",
    )
    parser.add_argument(
        "--no-slippery",
        action="store_true",
        help="Disable slippery ice (deterministic transitions)",
    )
    args = parser.parse_args()

    # Resolve slippery flag
    is_slippery = args.slippery and not args.no_slippery

    # Create environment adapter (user-provided code)
    adapter = FrozenLakeAdapter()

    # Default environment configuration
    default_config = {
        "map_name": args.map_name,
        "is_slippery": is_slippery,
        "render_mode": None,
    }

    # Create MCP server using framework
    server = MCPEnvironmentServer(
        adapter=adapter, server_name="FrozenLake-MCP", default_config=default_config
    )

    print("🏔️  FrozenLake MCP Server - Production Implementation")
    print(f"📡 Transport: {args.transport}")
    print(
        f"🎮 Environment: FrozenLake-v1 (map={args.map_name}, slippery={is_slippery})"
    )
    print()
    print("✅ ARCHITECTURE:")
    print("  • MCP Server: Pure FastMCP implementation")
    print("  • Framework: reward_kit.mcp utilities")
    print("  • Adapter: User-provided FrozenLake integration")
    print("  • Tools: Standard environment interface")
    print()

    # Run the server
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
