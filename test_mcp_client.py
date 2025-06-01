import asyncio
import logging
import json # Added import
from contextlib import AsyncExitStack

# Assuming the mcp library is installed in the environment
# and python-sdk is in a location where it can be imported,
# or that reward-kit's venv has access to it.
# If not, this import might need adjustment based on actual SDK setup.
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client # Corrected import
# from mcp.shared.message_types import TextContent # Removed import, will use duck typing

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration for the RewardKit Intermediary MCP Server
INTERMEDIARY_SERVER_URL = "http://localhost:8001/mcp" # Default from design doc

async def main():
    """
    Tests the initialize_session tool on the RewardKit Intermediary MCP Server.
    """
    logger.info(f"Attempting to connect to MCP server at {INTERMEDIARY_SERVER_URL}")

    # Define the payload for initialize_session
    init_session_payload = {
        "backends": [
            {
                "backend_name_ref": "filesystem_test", # Must match mcp_agent_config.yaml
                "num_instances": 1,
                # "template_id": "optional_template_for_filesystem", # Example
            },
            {
                "backend_name_ref": "memory_test", # Must match mcp_agent_config.yaml
                "num_instances": 1,
            },
        ]
    }
    
    rk_session_id = None # To store the session ID from the intermediary server

    async with AsyncExitStack() as stack:
        try:
            logger.info("Creating streamable HTTP transport...")
            # streamablehttp_client is an async context manager that should yield the streams.
            # Reverting to direct unpacking, as the type is confirmed to be a tuple.
            transport_tuple = await stack.enter_async_context(
                streamablehttp_client(INTERMEDIARY_SERVER_URL)
            )
            
            logger.info(f"Transport tuple type: {type(transport_tuple)}, length: {len(transport_tuple) if isinstance(transport_tuple, tuple) else 'N/A'}")
            if not isinstance(transport_tuple, tuple) or len(transport_tuple) != 3: # Expecting a 3-tuple now
                logger.error(f"Unexpected transport_tuple content: {transport_tuple}")
                raise ValueError(f"streamablehttp_client did not yield a 3-tuple as expected. Got: {type(transport_tuple)} with length {len(transport_tuple) if isinstance(transport_tuple, tuple) else 'N/A'}")

            # Unpack the first two elements for ClientSession
            read_stream, write_stream, _ = transport_tuple # Ignore the third element (get_session_id method)
            logger.info("Transport streams obtained from 3-tuple unpacking.")

            logger.info("Creating ClientSession...")
            # The ClientSession itself can take an initial session_id if we want to resume.
            # For a new session, we don't pass it.
            mcp_client_session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            logger.info("ClientSession created.")

            logger.info("Initializing ClientSession with server handshake...")
            await mcp_client_session.initialize() # Performs MCP handshake
            # The mcp_client_session.session_id attribute does not exist.
            # The session ID is managed internally by ClientSession after initialize().
            # Server logs confirm a session ID is created and used.
            logger.info(f"ClientSession handshake successful.") # Removed problematic log

            # 1. Call initialize_session directly on the Intermediary Server
            # Wrap payload under "args" key to match Pydantic model parameter name in server
            wrapped_init_payload = {"args": init_session_payload}
            logger.info(f"Calling 'initialize_session' with wrapped payload: {wrapped_init_payload}")
            init_session_result = await mcp_client_session.call_tool("initialize_session", wrapped_init_payload)
            logger.info(f"Raw 'initialize_session' result: {init_session_result}")

            if init_session_result.isError or not init_session_result.content or not hasattr(init_session_result.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if init_session_result.content and hasattr(init_session_result.content[0], "text"):
                    error_message = init_session_result.content[0].text
                elif init_session_result.isError:
                    error_message = "Tool call 'initialize_session' returned an error, but no text content part found."
                logger.error(f"Error from 'initialize_session': {error_message}")
                raise ValueError(f"Error from 'initialize_session': {error_message}")

            actual_init_result_dict = json.loads(init_session_result.content[0].text)
            logger.info(f"Parsed 'initialize_session' result: {actual_init_result_dict}")

            # Extract session_id and instance details from the result
            rk_session_id = actual_init_result_dict.get("rk_session_id")
            if not rk_session_id:
                raise ValueError("'initialize_session' did not return a 'rk_session_id'")
            
            logger.info(f"RewardKit Intermediary rk_session_id: {rk_session_id}")

            initialized_backends = actual_init_result_dict.get("initialized_backends", [])
            fs_instance_id = None
            for backend_res in initialized_backends:
                if backend_res.get("backend_name_ref") == "filesystem_test" and backend_res.get("instances"):
                    fs_instance_id = backend_res["instances"][0].get("instance_id")
                    break
            
            if not fs_instance_id:
                raise ValueError("Could not find instance_id for 'filesystem_test' in 'initialize_session' result.")
            logger.info(f"Filesystem_test instance_id: {fs_instance_id}")

            # 2. Call 'call_backend_tool' directly (e.g., list_files on filesystem_test)
            list_files_args = {
                "rk_session_id": rk_session_id,
                "backend_name_ref": "filesystem_test",
                "instance_id": fs_instance_id,
                "tool_name": "list_directory",  # Corrected tool name based on server logs
                "tool_args": {"path": "/data"}, # Explicitly list the served directory root
            }
            wrapped_list_files_payload = {"args": list_files_args}
            logger.info(f"Calling 'call_backend_tool' (list_directory) with wrapped payload: {wrapped_list_files_payload}")
            list_files_call_result = await mcp_client_session.call_tool("call_backend_tool", wrapped_list_files_payload)
            logger.info(f"Raw 'call_backend_tool' (list_directory) result: {list_files_call_result}")

            if list_files_call_result.isError or not list_files_call_result.content or not hasattr(list_files_call_result.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if list_files_call_result.content and hasattr(list_files_call_result.content[0], "text"):
                    error_message = list_files_call_result.content[0].text
                elif list_files_call_result.isError:
                    error_message = "Tool call 'call_backend_tool' (list_directory) returned an error, but no text content part found."
                logger.error(f"Error from 'call_backend_tool' (list_directory): {error_message}")
                raise ValueError(f"Error from 'call_backend_tool' (list_directory): {error_message}")
            
            actual_list_files_result_dict = json.loads(list_files_call_result.content[0].text)
            logger.info(f"Parsed 'call_backend_tool' (list_directory) result: {actual_list_files_result_dict}")
            
            # Example: Call 'read_graph' on memory_test backend
            mem_instance_id = None
            for backend_res in initialized_backends:
                if backend_res.get("backend_name_ref") == "memory_test" and backend_res.get("instances"):
                    mem_instance_id = backend_res["instances"][0].get("instance_id")
                    break
            if mem_instance_id:
                read_graph_args = { # Changed from ping_mem_args
                    "rk_session_id": rk_session_id,
                    "backend_name_ref": "memory_test",
                    "instance_id": mem_instance_id,
                    "tool_name": "read_graph", # Changed from "ping"
                    "tool_args": {}, # read_graph might take args, but empty for a basic test
                }
                wrapped_read_graph_payload = {"args": read_graph_args} # Changed variable name
                logger.info(f"Calling 'call_backend_tool' (read_graph memory_test) with wrapped payload: {wrapped_read_graph_payload}")
                read_graph_call_result = await mcp_client_session.call_tool("call_backend_tool", wrapped_read_graph_payload) # Changed variable name
                logger.info(f"Raw 'call_backend_tool' (read_graph memory_test) result: {read_graph_call_result}")

                if read_graph_call_result.isError or not read_graph_call_result.content or not hasattr(read_graph_call_result.content[0], "text"):
                    error_message = "Unknown error or non-text content"
                    if read_graph_call_result.content and hasattr(read_graph_call_result.content[0], "text"):
                        error_message = read_graph_call_result.content[0].text
                    elif read_graph_call_result.isError:
                        error_message = "Tool call 'call_backend_tool' (read_graph memory_test) returned an error, but no text content part found."
                    logger.error(f"Error from 'call_backend_tool' (read_graph memory_test): {error_message}")
                    raise ValueError(f"Error from 'call_backend_tool' (read_graph memory_test): {error_message}")

                actual_read_graph_result_dict = json.loads(read_graph_call_result.content[0].text) # Changed variable name
                logger.info(f"Parsed 'call_backend_tool' (read_graph memory_test) result: {actual_read_graph_result_dict}")

            # 3. Call 'cleanup_session' directly
            cleanup_session_args = {
                "rk_session_id": rk_session_id
            }
            wrapped_cleanup_payload = {"args": cleanup_session_args}
            # Removed mcp_client_session.session_id from log as it's not a public attribute
            logger.info(f"Calling 'cleanup_session' with rk_session_id: {rk_session_id}, wrapped payload: {wrapped_cleanup_payload}")
            cleanup_call_result = await mcp_client_session.call_tool("cleanup_session", wrapped_cleanup_payload)
            logger.info(f"Raw 'cleanup_session' result: {cleanup_call_result}")

            if cleanup_call_result.isError or not cleanup_call_result.content or not hasattr(cleanup_call_result.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if cleanup_call_result.content and hasattr(cleanup_call_result.content[0], "text"):
                    error_message = cleanup_call_result.content[0].text
                elif cleanup_call_result.isError:
                    error_message = "Tool call 'cleanup_session' returned an error, but no text content part found."
                logger.error(f"Error from 'cleanup_session': {error_message}")
                raise ValueError(f"Error from 'cleanup_session': {error_message}")

            actual_cleanup_result_dict = json.loads(cleanup_call_result.content[0].text)
            logger.info(f"Parsed 'cleanup_session' result: {actual_cleanup_result_dict}")
            
            # The server's cleanup_session returns the rk_session_id it operated on.
            if actual_cleanup_result_dict.get("rk_session_id") != rk_session_id: # Check against the rk_session_id we sent
                logger.warning(f"Cleanup rk_session_id mismatch: Expected {rk_session_id}, Got {actual_cleanup_result_dict.get('rk_session_id')} from cleanup tool")


        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            # If there's a specific MCP exception type, catch it
            # from mcp.shared.exceptions import MCPError
            # except MCPError as mcp_e:
            #     logger.error(f"MCP Error: {mcp_e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
