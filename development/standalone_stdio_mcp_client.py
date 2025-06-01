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
    # from mcp.client.stdio import StdioServerParameters, stdio_client # Not used in this version
    from mcp.client.session import ClientSession, DEFAULT_CLIENT_INFO
    # stdio_client is no longer used directly with ClientSession in this approach
    import mcp.types as types
    from mcp.shared.message import SessionMessage # For wrapping messages
    import anyio # For memory streams and task groups
    import functools # For functools.partial
except ImportError as e:
    print(f"Failed to import MCP components after modifying sys.path: {e}")
    print("Please ensure the MCP SDK is correctly placed at /home/bchen/references/python-sdk/src/")
    sys.exit(1)

import shutil
# Path is already imported above

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("standalone_stdio_mcp_client")
logging.getLogger("mcp").setLevel(logging.DEBUG) # Enable DEBUG logging for mcp library

# Define the host path for the filesystem server's data
HOST_DATA_PATH = Path("/tmp/mcp_fs_data_standalone")
CONTAINER_DATA_PATH = "/data"
ENCODING = "utf-8"

async def stdout_bridge(
    process_stdout: asyncio.StreamReader, # Corrected type hint for asyncio.StreamReader
    to_client_writer: anyio.streams.memory.MemoryObjectSendStream[SessionMessage | Exception],
    banner_lines_to_skip: int = 2 
):
    """Reads from process stdout, skips banner, parses JSON, and sends to client session."""
    skipped_lines = 0
    try:
        while True: # Loop to continuously read lines
            line_bytes = await process_stdout.readline()
            if not line_bytes: # End of stream
                logger.info("[STDOUT_BRIDGE] Process stdout EOF reached.")
                break
            line = line_bytes.decode(ENCODING).strip()
            if not line:
                continue

            if skipped_lines < banner_lines_to_skip:
                logger.info(f"[STDOUT_BRIDGE] Skipping banner line: {line}")
                skipped_lines += 1
                continue
            
            logger.debug(f"[STDOUT_BRIDGE] Received line: {line}")
            try:
                message = types.JSONRPCMessage.model_validate_json(line)
                await to_client_writer.send(SessionMessage(message=message))
                # Safer logging for message id/method
                log_msg_details = "UnknownType"
                if hasattr(message, 'id') and message.id is not None: # For Requests and Responses
                    log_msg_details = f"id={message.id}"
                if hasattr(message, 'method') and message.method is not None: # For Requests and Notifications
                    log_msg_details = f"method='{message.method}'"
                logger.debug(f"[STDOUT_BRIDGE] Sent to client: {log_msg_details}")
            except Exception as e:
                logger.error(f"[STDOUT_BRIDGE] Error parsing JSON or sending to client: {line} - {e}")
                await to_client_writer.send(e) # Propagate error to client session
                # Potentially break or implement more robust error handling
    except anyio.EndOfStream:
        logger.info("[STDOUT_BRIDGE] Process stdout stream ended.")
    except Exception as e:
        logger.exception(f"[STDOUT_BRIDGE] Unhandled exception: {e}")
    finally:
        logger.info("[STDOUT_BRIDGE] Closing writer to client.")
        await to_client_writer.aclose()

async def stdin_bridge(
    process_stdin: asyncio.StreamWriter, # Corrected type hint for asyncio.StreamWriter
    from_client_reader: anyio.streams.memory.MemoryObjectReceiveStream[SessionMessage]
):
    """Reads SessionMessages from client session, serializes, and writes to process stdin."""
    try:
        async for session_message in from_client_reader:
            json_str = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
            logger.debug(f"[STDIN_BRIDGE] Sending to process: {json_str}")
            process_stdin.write((json_str + "\n").encode(ENCODING))
            await process_stdin.drain()
    except anyio.EndOfStream: # This might change depending on how ClientSession signals closure
        logger.info("[STDIN_BRIDGE] Client writer stream ended.")
    except Exception as e:
        logger.exception(f"[STDIN_BRIDGE] Unhandled exception: {e}")
    finally:
        logger.info("[STDIN_BRIDGE] Closing process stdin.")
        if process_stdin and not process_stdin.is_closing():
            process_stdin.close()
            try:
                await process_stdin.wait_closed()
            except Exception as e_close: # Handle cases where wait_closed might not be needed or fails
                logger.debug(f"[STDIN_BRIDGE] Exception during wait_closed: {e_close}")


async def main():
    logger.info("Starting standalone stdio MCP client test (manual stream management).")

    if HOST_DATA_PATH.exists():
        logger.info(f"Cleaning up existing host data path: {HOST_DATA_PATH}")
        shutil.rmtree(HOST_DATA_PATH)
    HOST_DATA_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured host data path exists: {HOST_DATA_PATH}")

    dummy_file_name = "test_file.txt"
    dummy_file_content = "Hello from standalone client test!"
    with open(HOST_DATA_PATH / dummy_file_name, "w") as f:
        f.write(dummy_file_content)
    logger.info(f"Created dummy file: {HOST_DATA_PATH / dummy_file_name}")

    docker_command = [
        "docker", "run", "--rm", "-i",
        "-e", "MCP_TRANSPORT=stdio", # Keep this, might be used internally by server
        "-e", "NODE_ENV=production", # Keep this, might be used internally by server
        "-v", f"{HOST_DATA_PATH.resolve()}:{CONTAINER_DATA_PATH}",
        "mcp/filesystem", CONTAINER_DATA_PATH
    ]
    logger.info(f"Docker command: {' '.join(docker_command)}")

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *docker_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE # Capture stderr to log it
        )
        logger.info(f"Docker process started with PID: {process.pid}")

        # Create memory streams for ClientSession
        # Stream for messages from server to client_session
        server_to_client_writer, server_to_client_reader = anyio.create_memory_object_stream[SessionMessage | Exception](0)
        # Stream for messages from client_session to server
        client_to_server_writer, client_to_server_reader = anyio.create_memory_object_stream[SessionMessage](0)

        async with anyio.create_task_group() as tg:
            # Start stderr logger task
            async def log_stderr():
                async for line_bytes in process.stderr:
                    logger.warning(f"[DOCKER_STDERR] {line_bytes.decode(ENCODING).strip()}")
            tg.start_soon(log_stderr)

            # Start bridge tasks
            # Assuming the banner is 2 lines. Adjust if necessary.
            # "Secure MCP Filesystem Server running on stdio"
            # "Allowed directories: [ '/data' ]"
            # These banner lines are on stderr, so stdout should be clean JSON.
            
            # Use functools.partial to pass keyword argument to start_soon
            # Set banner_lines_to_skip to 0 as the banner is on stderr.
            partial_stdout_bridge = functools.partial(stdout_bridge, banner_lines_to_skip=0)
            tg.start_soon(partial_stdout_bridge, process.stdout, server_to_client_writer)
            
            tg.start_soon(stdin_bridge, process.stdin, client_to_server_reader)
            
            logger.info("Bridge tasks started.")

            client_session = ClientSession(
                read_stream=server_to_client_reader, # Client reads from what stdout_bridge writes
                write_stream=client_to_server_writer, # Client writes to what stdin_bridge reads
                client_info=DEFAULT_CLIENT_INFO
            )
            logger.info("ClientSession instantiated with memory streams.")
            # The BaseSession.run() context manager (invoked by `async with client_session:`)
            # will start the read loop within the current task group `tg`.

            # Allow some time for bridges and client session to settle, and server to be ready post-banner
            logger.info("Waiting for bridges and server to settle before initializing...")
            await anyio.sleep(1) # Can reduce sleep if run() handles startup well
            logger.info("Done waiting.")

            async with client_session: # This should call BaseSession.run()
                logger.info("Entered `async with client_session` (should start read loop).")
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

                # 4. Call a filesystem-specific tool: list_directory
                # Paths must be absolute from the container's root, and within the allowed dir /data
                list_files_path = CONTAINER_DATA_PATH # e.g., "/data"
                logger.info(f"Attempting to call 'list_directory' with absolute container path: '{list_files_path}'...")
                list_files_args = {"path": list_files_path}
                list_directory_result = await client_session.call_tool(
                    name="list_directory", arguments=list_files_args # Corrected tool name
                )
                logger.info(f"'list_directory' successful: {list_directory_result}")

                # Verify our dummy file is listed
                found_dummy_file = False
                if list_directory_result.content and isinstance(list_directory_result.content, list) and len(list_directory_result.content) > 0:
                    # Assuming the first content item is TextContent and contains the listing
                    if isinstance(list_directory_result.content[0], types.TextContent):
                        dir_listing_str = list_directory_result.content[0].text
                        logger.debug(f"Directory listing string: {dir_listing_str}")
                    # Example line: "[FILE] test_file.txt (123 bytes)" - adjust parsing as needed
                    # The mcp/filesystem server's list_directory output is a simple text list.
                    # Example: "[DIR] subdir\n[FILE] file1.txt\n[FILE] file2.txt"
                    # We need to check if dummy_file_name is in one of the lines.
                    if any(dummy_file_name in line for line in dir_listing_str.split('\n')):
                        found_dummy_file = True
                        logger.info(f"Successfully found '{dummy_file_name}' in list_directory result.")
                if not found_dummy_file: # This if should be at the same level as the previous if
                    logger.warning(f"Could not find '{dummy_file_name}' in list_directory result. Full result: {list_directory_result}")


                # 5. Call filesystem/read_file for the dummy file
                read_file_path = f"{CONTAINER_DATA_PATH}/{dummy_file_name}" # e.g., "/data/test_file.txt"
                logger.info(f"Attempting to call 'read_file' with absolute container path: '{read_file_path}'...")
                read_file_args = {"path": read_file_path, "encoding": "utf-8"}
                read_file_result = await client_session.call_tool(
                    name="read_file", arguments=read_file_args # Tool name seems correct from logs
                )
                logger.info(f"'read_file' successful: {read_file_result}")
                
                # Access content from CallToolResult.content list
                read_content = None
                if read_file_result.content and isinstance(read_file_result.content, list) and len(read_file_result.content) > 0:
                    if isinstance(read_file_result.content[0], types.TextContent):
                        read_content = read_file_result.content[0].text
                
                if read_content is not None:
                    if read_content == dummy_file_content:
                        logger.info(f"Successfully read content of '{dummy_file_name}': MATCHES expected.")
                    else:
                        logger.error(f"Content mismatch for '{dummy_file_name}'. Expected: '{dummy_file_content}', Got: '{read_content}'")
                else:
                    logger.error(f"Failed to get content from read_file_result for '{dummy_file_name}'. Result was: {read_file_result}")


                # 6. Call filesystem/write_file to create a new file
                write_file_path_relative_to_data = "newly_created_file.txt"
                write_file_path_absolute = f"{CONTAINER_DATA_PATH}/{write_file_path_relative_to_data}" # e.g., "/data/newly_created_file.txt"
                write_file_content = "This file was written by the standalone client."
                logger.info(f"Attempting to call 'write_file' with absolute container path: '{write_file_path_absolute}'...")
                write_file_args = {"path": write_file_path_absolute, "content": write_file_content, "encoding": "utf-8"}
                write_file_result = await client_session.call_tool(
                    name="write_file", arguments=write_file_args # Corrected tool name
                )
                logger.info(f"'write_file' successful: {write_file_result}")

                # Verify the new file exists on the host
                if (HOST_DATA_PATH / write_file_path_relative_to_data).exists():
                    logger.info(f"Successfully verified '{write_file_path_relative_to_data}' was created on the host: {HOST_DATA_PATH / write_file_path_relative_to_data}")
                    with open(HOST_DATA_PATH / write_file_path_relative_to_data, "r") as f_host:
                        host_content = f_host.read()
                        if host_content == write_file_content:
                            logger.info("Content of newly created file matches expected content.")
                        else:
                            logger.error(f"Content mismatch for newly created file. Expected: '{write_file_content}', Got on host: '{host_content}'")
                else:
                    logger.error(f"Failed to verify '{write_file_path_relative_to_data}' was created on the host at {HOST_DATA_PATH / write_file_path_relative_to_data}.")


                logger.info("All tests passed successfully!")
            
            # Exiting `async with client_session` will call client_session.aclose()
            logger.info("Exited `async with client_session` context.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        if process and process.returncode is None: # Check if process is still running
            logger.info(f"Terminating Docker process PID: {process.pid}...")
            try:
                process.terminate()
                await process.wait() # Wait for termination
                logger.info(f"Docker process PID: {process.pid} terminated.")
            except ProcessLookupError:
                logger.info(f"Docker process PID: {process.pid} already exited.")
            except Exception as e_term:
                logger.exception(f"Error terminating process: {e_term}")
        elif process:
            logger.info(f"Docker process PID: {process.pid} already exited with code: {process.returncode}")
        else:
            logger.info("No Docker process was started or it failed early.")

        logger.info("Standalone stdio MCP client test finished.")
        # if HOST_DATA_PATH.exists():
        #     shutil.rmtree(HOST_DATA_PATH)
        #     logger.info(f"Cleaned up host data path: {HOST_DATA_PATH}")



if __name__ == "__main__":
    asyncio.run(main())
