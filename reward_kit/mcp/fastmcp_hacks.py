# reward_kit/mcp/fastmcp_hacks.py

"""
Workarounds for limitations in the FastMCP library.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, cast

from mcp.server.fastmcp.resources import FunctionResource

# Forward declare to avoid circular import, for type hinting only
if False:
    from mcp.server.fastmcp.server import FastMCP


class FunctionResourceWithContext(FunctionResource):
    """
    A custom FunctionResource that correctly injects the FastMCP Context
    into the resource handler function.
    """

    _fastmcp_server: "FastMCP"

    def __init__(
        self, *, fn: Callable[..., Any], fastmcp_server: "FastMCP", **kwargs: Any
    ):
        """Initializes the custom resource, storing a reference to the main server instance."""
        super().__init__(fn=fn, **kwargs)
        self._fastmcp_server = fastmcp_server

    async def read(self, **kwargs: Any) -> str | bytes | dict[str, Any]:
        """
        Overrides the default read method to inject the context.
        """
        # Manually get the context for the current request from the server instance.
        ctx = self._fastmcp_server.get_context()

        # Check if the handler function expects a context argument
        sig = inspect.signature(self.fn)
        if "ctx" in sig.parameters:
            # If so, call it with the context
            result = await self.fn(ctx=ctx, **kwargs)
        else:
            # Otherwise, call it normally
            result = await self.fn(**kwargs)

        return cast(str | bytes | dict[str, Any], result)
