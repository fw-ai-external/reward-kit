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

            # 1. Call initialize_session on the Intermediary Server via the proxy tool
            proxied_init_payload = {
                "actual_tool_name": "initialize_session",
                "actual_tool_args": init_session_payload
            }
            logger.info(f"Calling 'execute_proxied_tool' for 'initialize_session' with payload: {proxied_init_payload}")
            # The result from execute_proxied_tool will be the result from initialize_session_actual
            init_result_proxied = await mcp_client_session.call_tool("execute_proxied_tool", proxied_init_payload)
            logger.info(f"Raw 'execute_proxied_tool' for 'initialize_session' result: {init_result_proxied}")

            if init_result_proxied.isError or not init_result_proxied.content or not hasattr(init_result_proxied.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if init_result_proxied.content and hasattr(init_result_proxied.content[0], "text"):
                    error_message = init_result_proxied.content[0].text
                elif init_result_proxied.isError:
                    error_message = "Tool call returned an error, but no text content part found."

                logger.error(f"Error from 'initialize_session' proxy: {error_message}")
                raise ValueError(f"Error from 'initialize_session' proxy: {error_message}")

            actual_init_result_dict = json.loads(init_result_proxied.content[0].text)
            logger.info(f"Parsed actual 'initialize_session' result: {actual_init_result_dict}")

            # Extract session_id and instance details from the result (which is the actual init_result)
            # The key in the JSON response is "rk_session_id" as per server logs and design intent.
            rk_session_id = actual_init_result_dict.get("rk_session_id")
            if not rk_session_id:
                raise ValueError("'initialize_session' (proxied) did not return a 'rk_session_id' in its actual result")
            
            logger.info(f"RewardKit Intermediary rk_session_id (from proxied call): {rk_session_id}")

            initialized_backends = actual_init_result_dict.get("initialized_backends", [])
            fs_instance_id = None
            for backend_res in initialized_backends:
                # Accessing dictionary keys directly now
                if backend_res.get("backend_name_ref") == "filesystem_test" and backend_res.get("instances"):
                    fs_instance_id = backend_res["instances"][0].get("instance_id")
                    break
            
            if not fs_instance_id:
                raise ValueError("Could not find instance_id for 'filesystem_test' in 'initialize_session' (proxied) result.")
            logger.info(f"Filesystem_test instance_id: {fs_instance_id}")

            # 2. Call call_backend_tool (e.g., list_files on filesystem_test) via proxy
            list_files_args = {
                "rk_session_id": rk_session_id, # Pass the session ID
                "backend_name_ref": "filesystem_test",
                "instance_id": fs_instance_id,
                "tool_name": "list_files", 
                "tool_args": {"path": "/"}, 
            }
            proxied_list_files_payload = {
                "actual_tool_name": "call_backend_tool",
                "actual_tool_args": list_files_args
            }
            logger.info(f"Calling 'execute_proxied_tool' for 'call_backend_tool' (list_files) with payload: {proxied_list_files_payload}")
            list_files_call_result = await mcp_client_session.call_tool("execute_proxied_tool", proxied_list_files_payload)
            logger.info(f"Raw 'execute_proxied_tool' for 'call_backend_tool' (list_files) result: {list_files_call_result}")

            if list_files_call_result.isError or not list_files_call_result.content or not hasattr(list_files_call_result.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if list_files_call_result.content and hasattr(list_files_call_result.content[0], "text"):
                    error_message = list_files_call_result.content[0].text
                elif list_files_call_result.isError:
                    error_message = "Tool call (list_files) returned an error, but no text content part found."
                logger.error(f"Error from 'call_backend_tool' (list_files) proxy: {error_message}")
                raise ValueError(f"Error from 'call_backend_tool' (list_files) proxy: {error_message}")
            
            actual_list_files_result_dict = json.loads(list_files_call_result.content[0].text)
            logger.info(f"Parsed actual 'call_backend_tool' (list_files) result: {actual_list_files_result_dict}")
            
            # Example: Ping the memory_test backend via proxy
            mem_instance_id = None
            for backend_res in initialized_backends:
                if backend_res.get("backend_name_ref") == "memory_test" and backend_res.get("instances"):
                    mem_instance_id = backend_res["instances"][0].get("instance_id")
                    break
            if mem_instance_id:
                ping_mem_args = {
                    "rk_session_id": rk_session_id, # Pass the session ID
                    "backend_name_ref": "memory_test",
                    "instance_id": mem_instance_id,
                    "tool_name": "ping", 
                    "tool_args": {},
                }
                proxied_ping_mem_payload = {
                    "actual_tool_name": "call_backend_tool",
                    "actual_tool_args": ping_mem_args
                }
                logger.info(f"Calling 'execute_proxied_tool' for 'call_backend_tool' (ping memory_test) with payload: {proxied_ping_mem_payload}")
                ping_call_result = await mcp_client_session.call_tool("execute_proxied_tool", proxied_ping_mem_payload)
                logger.info(f"Raw 'execute_proxied_tool' for 'call_backend_tool' (ping memory_test) result: {ping_call_result}")

                if ping_call_result.isError or not ping_call_result.content or not hasattr(ping_call_result.content[0], "text"):
                    error_message = "Unknown error or non-text content"
                    if ping_call_result.content and hasattr(ping_call_result.content[0], "text"):
                        error_message = ping_call_result.content[0].text
                    elif ping_call_result.isError:
                        error_message = "Tool call (ping memory_test) returned an error, but no text content part found."
                    logger.error(f"Error from 'call_backend_tool' (ping memory_test) proxy: {error_message}")
                    raise ValueError(f"Error from 'call_backend_tool' (ping memory_test) proxy: {error_message}")

                actual_ping_result_dict = json.loads(ping_call_result.content[0].text)
                logger.info(f"Parsed actual 'call_backend_tool' (ping memory_test) result: {actual_ping_result_dict}")


            # 3. Call cleanup_session on the Intermediary Server via proxy
            cleanup_session_args = {
                "rk_session_id": rk_session_id # Pass the session ID
            }
            proxied_cleanup_payload = {
                "actual_tool_name": "cleanup_session",
                "actual_tool_args": cleanup_session_args 
            }
            # Note: mcp_client_session.session_id is the transport session_id.
            # The rk_session_id is the one returned by our initialize_session logic and now passed to cleanup_session.
            logger.info(f"Calling 'execute_proxied_tool' for 'cleanup_session' with rk_session_id: {rk_session_id} (transport session: {mcp_client_session.session_id})")
            cleanup_call_result = await mcp_client_session.call_tool("execute_proxied_tool", proxied_cleanup_payload)
            logger.info(f"Raw 'execute_proxied_tool' for 'cleanup_session' result: {cleanup_call_result}")

            if cleanup_call_result.isError or not cleanup_call_result.content or not hasattr(cleanup_call_result.content[0], "text"):
                error_message = "Unknown error or non-text content"
                if cleanup_call_result.content and hasattr(cleanup_call_result.content[0], "text"):
                    error_message = cleanup_call_result.content[0].text
                elif cleanup_call_result.isError:
                    error_message = "Tool call (cleanup_session) returned an error, but no text content part found."
                logger.error(f"Error from 'cleanup_session' proxy: {error_message}")
                raise ValueError(f"Error from 'cleanup_session' proxy: {error_message}")

            actual_cleanup_result_dict = json.loads(cleanup_call_result.content[0].text)
            logger.info(f"Parsed actual 'cleanup_session' result: {actual_cleanup_result_dict}")
            
            # The session_id returned by cleanup_session_actual is the transport_session_id it operated on.
            if actual_cleanup_result_dict.get("session_id") != mcp_client_session.session_id:
                logger.warning(f"Cleanup session ID mismatch: Expected transport session {mcp_client_session.session_id}, Got {actual_cleanup_result_dict.get('session_id')} from cleanup tool")


        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            # If there's a specific MCP exception type, catch it
            # from mcp.shared.exceptions import MCPError
            # except MCPError as mcp_e:
            #     logger.error(f"MCP Error: {mcp_e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
