import asyncio
import logging
import signal

import click # Using click for CLI arguments
import yaml # For loading config file

from reward_kit.mcp_agent.config import AppConfig
from reward_kit.mcp_agent.intermediary_server import RewardKitIntermediaryServer

logger = logging.getLogger(__name__)

# Global server instance to be managed by signal handlers
server_instance: Optional[RewardKitIntermediaryServer] = None

async def main_async(config_path: str, host: str, port: int):
    """
    Asynchronous main function to load config, start the server,
    and handle graceful shutdown.
    """
    global server_instance
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
        app_config = AppConfig(**raw_config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return
    except Exception as e: # Catch Pydantic validation errors etc.
        logger.error(f"Error loading or validating AppConfig from {config_path}: {e}")
        return

    logging.basicConfig(level=app_config.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Configuration loaded from {config_path}")

    server_instance = RewardKitIntermediaryServer(app_config=app_config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))

    try:
        # This is a placeholder for how the MCP server would actually run.
        # The real MCP SDK would provide a way to run the server, e.g., server.run(host, port)
        # For now, we'll call our server's start and then keep the event loop running.
        await server_instance.start()
        logger.info(f"RewardKit Intermediary MCP Server attempting to listen on {host}:{port} (actual binding depends on MCP SDK)")
        
        # Keep the server running until a shutdown signal is received.
        # In a real ASGI server context (like Uvicorn), this loop is managed by the ASGI server.
        # Since we are using a placeholder BaseMcpServer, we simulate this.
        while True: # Keep alive until shutdown signal
            if server_instance is None: # Check if shutdown has been triggered
                logger.info("Server instance is None, exiting main_async loop.")
                break
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Main task cancelled, initiating shutdown.")
    except Exception as e:
        logger.error(f"An error occurred during server operation: {e}", exc_info=True)
    finally:
        if server_instance: # Ensure shutdown is called if not already None
            logger.info("Ensuring server shutdown in finally block.")
            await server_instance.stop()
            logger.info("Server shutdown process completed in finally block.")
        else:
            logger.info("Server already shut down or was not started.")


async def shutdown(sig: signal.Signals, loop: asyncio.AbstractEventLoop):
    """Graceful shutdown handler."""
    global server_instance
    logger.warning(f"Received shutdown signal: {sig.name}. Initiating graceful shutdown...")

    if server_instance:
        await server_instance.stop()
        server_instance = None # Indicate server has been stopped

    # Optional: Cancel all other running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All outstanding tasks cancelled.")
    
    # loop.stop() # This might be called by the ASGI server in a real scenario


@click.command()
@click.option(
    "--config",
    "config_path",
    default="mcp_agent_config.yaml",
    help="Path to the YAML configuration file for the MCP agent server.",
    type=click.Path(exists=False), # exists=True would fail if file not there, but we handle FileNotFoundError
)
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--port", default=8000, help="Port to bind the server to.")
def main_cli(config_path: str, host: str, port: int):
    """
    CLI entry point to run the RewardKit Intermediary MCP Server.
    """
    try:
        asyncio.run(main_async(config_path, host, port))
    except KeyboardInterrupt:
        logger.info("CLI interrupted by KeyboardInterrupt. Exiting.")
    finally:
        logger.info("MCP Agent Server CLI finished.")

if __name__ == "__main__":
    main_cli()
