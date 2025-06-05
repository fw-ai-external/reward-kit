import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

logger = logging.getLogger(__name__)

# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="Docker CLI not available",
)


class MCPServerManager:
    """Helper class to manage MCP intermediary server for tests."""

    def __init__(self, config_path: str, host: str = "localhost", port: int = 8001):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.server_url = f"http://{host}:{port}"

    async def start_server(self, timeout: int = 30) -> None:
        """Start the MCP intermediary server."""
        cmd = [
            ".venv/bin/python",
            "reward_kit/mcp_agent/main.py",
            "--config",
            self.config_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        logger.info(f"Starting MCP server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=1)
                if response.status_code == 200:
                    logger.info("MCP server is ready")
                    return
            except requests.RequestException:
                pass

            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"MCP server process exited unexpectedly:\n"
                    f"stdout: {stdout}\n"
                    f"stderr: {stderr}"
                )

            await asyncio.sleep(0.5)

        raise TimeoutError(f"MCP server failed to start within {timeout} seconds")

    def stop_server(self) -> None:
        """Stop the MCP intermediary server."""
        if self.process:
            logger.info("Stopping MCP server")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing")
                self.process.kill()
                self.process.wait()

            self.process = None


@pytest.fixture(scope="function")
async def mcp_server_manager():
    """Fixture that provides an MCP server manager."""
    config_path = "mcp_agent_config.yaml"

    # Ensure config file exists
    if not os.path.exists(config_path):
        pytest.skip(f"MCP config file {config_path} not found")

    manager = MCPServerManager(config_path)
    yield manager

    # Cleanup
    manager.stop_server()


@pytest.mark.asyncio
@pytest.mark.docker
async def test_mcp_intermediary_server_startup(mcp_server_manager: MCPServerManager):
    """Test that the MCP intermediary server can start up successfully."""
    await mcp_server_manager.start_server()

    # Verify server is responding
    response = requests.get(f"{mcp_server_manager.server_url}/health")
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.docker
async def test_filesystem_rl_scenario_integration(mcp_server_manager: MCPServerManager):
    """Test the filesystem RL scenario through the MCP intermediary server."""
    await mcp_server_manager.start_server()

    # Run the RL filesystem scenario test
    from tests.mcp_agent.test_rl_filesystem_scenario import main as run_scenario

    # This should not raise an exception if the test passes
    await run_scenario()


@pytest.mark.asyncio
@pytest.mark.docker
@pytest.mark.slow
async def test_mcp_agent_cli_integration(mcp_server_manager: MCPServerManager):
    """Test running the MCP agent filesystem RL example via the CLI."""
    await mcp_server_manager.start_server()

    # Check if the example config exists
    example_config_dir = Path("examples/mcp_agent_filesystem_rl")
    if not example_config_dir.exists():
        pytest.skip(f"Example config directory {example_config_dir} not found")

    # Run the CLI command
    cmd = [
        ".venv/bin/python",
        "-m",
        "reward_kit.cli",
        "run",
        "--config-path",
        str(example_config_dir),
        "--config-name",
        "config",
    ]

    logger.info(f"Running CLI command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
    )

    if result.returncode != 0:
        logger.error(f"CLI command failed with exit code {result.returncode}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")

    assert result.returncode == 0, f"CLI command failed: {result.stderr}"


@pytest.mark.asyncio
@pytest.mark.docker
async def test_mcp_server_health_endpoint(mcp_server_manager: MCPServerManager):
    """Test that the MCP server health endpoint works correctly."""
    await mcp_server_manager.start_server()

    response = requests.get(f"{mcp_server_manager.server_url}/health")
    assert response.status_code == 200

    # Check if response contains expected health information
    health_data = response.json()
    assert "status" in health_data
    assert health_data["status"] in ["healthy", "ok"]
