import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the MCP SDK path to sys.path
# Assuming the SDK is in /home/bchen/references/python-sdk/src/
# and this script needs to import 'mcp'
SDK_SRC_PATH = Path("/home/bchen/references/python-sdk/src/")
if SDK_SRC_PATH.is_dir() and str(SDK_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_SRC_PATH))
    print(f"Added {SDK_SRC_PATH} to sys.path")

# Now try to import mcp components
try:
    import mcp.types as types
    from mcp.client.session import DEFAULT_CLIENT_INFO, ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError as e:
    print(f"Failed to import MCP components after modifying sys.path: {e}")
    print(
        "Please ensure the MCP SDK is correctly placed at /home/bchen/references/python-sdk/src/"
    )
    sys.exit(1)

import shutil

# Path is already imported above

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("standalone_stdio_mcp_client")

# Define the host path for the filesystem server's data
# This directory will be created if it doesn't exist
# and mounted into the Docker container.
HOST_DATA_PATH = Path("/tmp/mcp_fs_data_standalone")
CONTAINER_DATA_PATH = "/data"


async def main():
    logger.info("Starting standalone stdio MCP client test.")

    # Ensure the host data path exists
    if HOST_DATA_PATH.exists():
        # Clean up previous run's data for a fresh start
        logger.info(f"Cleaning up existing host data path: {HOST_DATA_PATH}")
        shutil.rmtree(HOST_DATA_PATH)
    HOST_DATA_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured host data path exists: {HOST_DATA_PATH}")

    # Create a dummy file in the host path to test read operations
    dummy_file_name = "test_file.txt"
    dummy_file_content = "Hello from standalone client test!"
    with open(HOST_DATA_PATH / dummy_file_name, "w") as f:
        f.write(dummy_file_content)
    logger.info(f"Created dummy file: {HOST_DATA_PATH / dummy_file_name}")

    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "--rm",  # Remove container on exit
            "-i",  # Keep STDIN open even if not attached
            "-v",
            f"{HOST_DATA_PATH.resolve()}:{CONTAINER_DATA_PATH}",  # Mount host data path
            "mcp/filesystem",  # The Docker image for the filesystem server
            # The "stdio" argument was causing the server to look for a "stdio" directory.
            # The mcp/filesystem server likely uses /data as its main argument (from mcp_agent_config.yaml)
            # and might default to stdio or use an env var. For now, let's pass only /data.
            CONTAINER_DATA_PATH,  # Argument to the mcp/filesystem server: the path to serve
        ],
        # cwd=str(Path.home()), # Optional: if the command needs a specific CWD
        env=os.environ.copy(),  # Pass current environment
    )

    logger.info(
        f"StdioServerParameters configured: command='{server_params.command}', args='{' '.join(server_params.args)}'"
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            logger.info("Stdio client transport established.")

            client_session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                client_info=DEFAULT_CLIENT_INFO,  # Use default client info
            )
            logger.info("ClientSession instantiated.")

            # 1. Initialize
            logger.info("Attempting to initialize session...")
            init_result = await client_session.initialize()
            logger.info(f"Session initialized successfully: {init_result}")

            # 2. List Tools
            logger.info("Attempting to list tools...")
            tools_result = await client_session.list_tools()
            logger.info(f"Tools listed successfully: {tools_result}")

            # 3. Ping
            logger.info("Attempting to ping server...")
            ping_result = await client_session.send_ping()
            logger.info(f"Ping successful: {ping_result}")

            # 4. Call a filesystem-specific tool: list_files
            list_files_path = (
                "/"  # List root directory inside the container's data mount
            )
            logger.info(
                f"Attempting to call 'filesystem/list_files' with path: '{list_files_path}'..."
            )
            list_files_args = {"path": list_files_path}
            list_files_result = await client_session.call_tool(
                name="filesystem/list_files", arguments=list_files_args
            )
            logger.info(f"'filesystem/list_files' successful: {list_files_result}")

            # Verify our dummy file is listed
            found_dummy_file = False
            if list_files_result.root and list_files_result.root.files:
                for file_info in list_files_result.root.files:
                    if file_info.name == dummy_file_name:
                        found_dummy_file = True
                        logger.info(
                            f"Successfully found '{dummy_file_name}' in list_files result."
                        )
                        break
            if not found_dummy_file:
                logger.warning(
                    f"Could not find '{dummy_file_name}' in list_files result."
                )

            # 5. Call filesystem/read_file for the dummy file
            read_file_path = f"/{dummy_file_name}"  # Path inside the container
            logger.info(
                f"Attempting to call 'filesystem/read_file' with path: '{read_file_path}'..."
            )
            read_file_args = {"path": read_file_path, "encoding": "utf-8"}
            read_file_result = await client_session.call_tool(
                name="filesystem/read_file", arguments=read_file_args
            )
            logger.info(f"'filesystem/read_file' successful.")

            if read_file_result.root and read_file_result.root.content:
                if read_file_result.root.content == dummy_file_content:
                    logger.info(
                        f"Successfully read content of '{dummy_file_name}': MATCHES expected."
                    )
                else:
                    logger.error(
                        f"Content mismatch for '{dummy_file_name}'. Expected: '{dummy_file_content}', Got: '{read_file_result.root.content}'"
                    )
            else:
                logger.error(
                    f"Failed to get content from read_file_result for '{dummy_file_name}'."
                )

            # 6. Call filesystem/write_file to create a new file
            write_file_path = "/newly_created_file.txt"
            write_file_content = "This file was written by the standalone client."
            logger.info(
                f"Attempting to call 'filesystem/write_file' with path: '{write_file_path}'..."
            )
            write_file_args = {
                "path": write_file_path,
                "content": write_file_content,
                "encoding": "utf-8",
            }
            write_file_result = await client_session.call_tool(
                name="filesystem/write_file", arguments=write_file_args
            )
            logger.info(f"'filesystem/write_file' successful: {write_file_result}")

            # Verify the new file exists on the host
            if (HOST_DATA_PATH / "newly_created_file.txt").exists():
                logger.info(
                    f"Successfully verified '{write_file_path}' was created on the host: {HOST_DATA_PATH / 'newly_created_file.txt'}"
                )
                with open(HOST_DATA_PATH / "newly_created_file.txt", "r") as f_host:
                    host_content = f_host.read()
                    if host_content == write_file_content:
                        logger.info(
                            "Content of newly created file matches expected content."
                        )
                    else:
                        logger.error(
                            f"Content mismatch for newly created file. Expected: '{write_file_content}', Got on host: '{host_content}'"
                        )
            else:
                logger.error(
                    f"Failed to verify '{write_file_path}' was created on the host."
                )

            logger.info("All tests passed successfully!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        logger.info("Standalone stdio MCP client test finished.")
        # Optional: Clean up the host data path after test
        # if HOST_DATA_PATH.exists():
        #     shutil.rmtree(HOST_DATA_PATH)
        #     logger.info(f"Cleaned up host data path: {HOST_DATA_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
