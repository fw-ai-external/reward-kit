import asyncio
import json  # For MCP payloads
import logging
import os  # For creating temp dir
import select  # For non-blocking read with timeout
import shutil  # For removing temp dir
import socket  # For socket.SHUT_WR
import time  # For unique temp dir names
import uuid  # Make sure uuid is imported at the top

import docker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DOCKER_IMAGE_TO_TEST = "mcp/filesystem"  # Can be changed to "mcp/memory"
CONTAINER_NAME_PREFIX = "direct-run-test-"

# Configuration specific to the image being tested
IMAGE_CONFIG = {
    "mcp/filesystem": {
        "command_args": ["/data"],
        "volume_mount": {
            "host_path_base": "/tmp/test_docker_run_data",  # Base for unique temp dir
            "container_path": "/data",
            "mode": "rw",
        },
        "test_mcp_tool": "list_tools",
        "test_mcp_args": {},
    },
    "mcp/memory": {
        "command_args": [],  # No specific command args for memory server
        "volume_mount": None,  # No volume mount needed for memory server
        "test_mcp_tool": "list_tools",
        "test_mcp_args": {},
    },
}


async def main():
    client = None
    container = None
    temp_host_dir = None
    try:
        logger.info("Initializing Docker client...")
        client = docker.from_env()
        logger.info("Docker client initialized.")

        logger.info(f"Pinging Docker daemon...")
        if not client.ping():
            logger.error("Docker daemon ping failed.")
            return
        logger.info("Docker daemon ping successful.")

        try:
            logger.info(f"Pulling image {DOCKER_IMAGE_TO_TEST} if not present...")
            client.images.pull(DOCKER_IMAGE_TO_TEST)
            logger.info(f"Image {DOCKER_IMAGE_TO_TEST} is available.")
        except docker.errors.ImageNotFound:
            logger.error(
                f"Image {DOCKER_IMAGE_TO_TEST} not found. Please ensure it's available or build it."
            )
            return
        except docker.errors.APIError as e:
            logger.error(f"APIError pulling image {DOCKER_IMAGE_TO_TEST}: {e}")
            return

        container_name = f"{CONTAINER_NAME_PREFIX}{DOCKER_IMAGE_TO_TEST.replace('/', '-')}-{uuid.uuid4().hex[:8]}"

        current_image_config = IMAGE_CONFIG.get(DOCKER_IMAGE_TO_TEST)
        if not current_image_config:
            logger.error(
                f"No configuration found for image {DOCKER_IMAGE_TO_TEST}. Please add it to IMAGE_CONFIG."
            )
            return

        command_args = current_image_config["command_args"]
        volume_mount_config = current_image_config["volume_mount"]
        volumes_arg = {}

        if volume_mount_config:
            # Create a unique temp directory on the host for this run
            unique_suffix = f"{DOCKER_IMAGE_TO_TEST.replace('/', '-')}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
            temp_host_dir = os.path.join(
                volume_mount_config["host_path_base"], unique_suffix
            )
            os.makedirs(temp_host_dir, exist_ok=True)
            logger.info(
                f"Created temporary host directory for volume mount: {temp_host_dir}"
            )
            volumes_arg = {
                temp_host_dir: {
                    "bind": volume_mount_config["container_path"],
                    "mode": volume_mount_config["mode"],
                }
            }

        logger.info(
            f"Attempting to run container '{container_name}' from image '{DOCKER_IMAGE_TO_TEST}' with command args: {command_args}, volumes: {volumes_arg}"
        )

        container = client.containers.run(
            DOCKER_IMAGE_TO_TEST,
            command=command_args,
            detach=True,
            name=container_name,
            volumes=volumes_arg if volumes_arg else None,
            stdin_open=True,  # Crucial for stdio interaction
            # tty=False, # Usually False for direct stdio stream manipulation
        )

        logger.info(
            f"client.containers.run() call completed for '{container_name}'. Container ID: {container.id}"
        )

        # Brief pause for container to initialize
        await asyncio.sleep(2)

        refetched_container = client.containers.get(container.id)
        logger.info(
            f"Status of container '{refetched_container.name}' after init: {refetched_container.status}"
        )

        if refetched_container.status == "running":
            logger.info(f"Attempting MCP call to container {container.id} via stdio...")

            mcp_request = {
                "request_id": f"test-req-{uuid.uuid4().hex[:6]}",
                "tool_name": current_image_config["test_mcp_tool"],
                "arguments": current_image_config["test_mcp_args"],
            }
            request_str = json.dumps(mcp_request) + "\n"

            try:
                # Attach to the container's stdio
                # stream=True gives raw stream, logs=False to avoid log stream mixed with stdio
                attached_socket = refetched_container.attach_socket(
                    params={
                        "stdin": True,
                        "stdout": True,
                        "stderr": True,
                        "stream": True,
                        "logs": False,
                    }
                )

                # Send request
                logger.info(f"Sending to stdin: {request_str.strip()}")
                if hasattr(
                    attached_socket, "_sock"
                ):  # Access underlying socket if available (docker-py specific)
                    sock_fd = attached_socket._sock
                    sock_fd.sendall(request_str.encode("utf-8"))
                    try:
                        sock_fd.shutdown(socket.SHUT_WR)  # Signal EOF to server's stdin
                    except OSError as e:
                        logger.warning(
                            f"OSError during sock_fd.shutdown(SHUT_WR): {e}. Server might have already closed connection."
                        )
                else:  # Fallback for other socket-like objects, less reliable for shutdown
                    # This path is less likely if attach_socket(stream=True) returns a raw socket wrapper
                    attached_socket.sendall(request_str.encode("utf-8"))
                    if hasattr(attached_socket, "shutdown") and callable(
                        getattr(attached_socket, "shutdown")
                    ):
                        try:
                            attached_socket.shutdown(socket.SHUT_WR)
                        except OSError as e:
                            logger.warning(
                                f"OSError during attached_socket.shutdown(SHUT_WR): {e}."
                            )

                # Read response (this is a simplified read, might need more robust line/buffer handling)
                response_buffer = b""
                timeout_seconds = 5  # 5 second timeout for response
                start_time = time.time()

                # Set socket to non-blocking to implement timeout
                if hasattr(attached_socket, "_sock"):
                    attached_socket._sock.setblocking(False)

                while True:
                    if time.time() - start_time > timeout_seconds:
                        logger.error("Timeout waiting for MCP response.")
                        break

                    ready_to_read, _, _ = select.select(
                        [
                            (
                                attached_socket._sock
                                if hasattr(attached_socket, "_sock")
                                else attached_socket
                            )
                        ],
                        [],
                        [],
                        0.1,
                    )  # 0.1s select timeout

                    if ready_to_read:
                        sock_to_read_from = ready_to_read[0]
                        try:
                            # Docker multiplexes stdout/stderr. Header is 8 bytes: 1 byte type (0=stdin, 1=stdout, 2=stderr), 3 bytes padding, 4 bytes size.
                            header = sock_to_read_from.recv(8)
                            if not header:
                                break  # Connection closed

                            stream_type = header[0]
                            payload_size = int.from_bytes(header[4:], byteorder="big")

                            payload = b""
                            while len(payload) < payload_size:
                                chunk = sock_to_read_from.recv(
                                    payload_size - len(payload)
                                )
                                if not chunk:
                                    raise ConnectionError(
                                        "Socket closed prematurely while reading payload"
                                    )
                                payload += chunk

                            if stream_type == 1:  # stdout
                                response_buffer += payload
                                if (
                                    b"\n" in response_buffer
                                ):  # Assuming newline delimited JSON
                                    break
                            elif stream_type == 2:  # stderr
                                logger.warning(
                                    f"STDERR from container: {payload.decode('utf-8', errors='replace').strip()}"
                                )
                        except BlockingIOError:
                            await asyncio.sleep(0.05)  # Short sleep if no data
                            continue
                        except (
                            ConnectionError
                        ) as e:  # Handle connection closed by server
                            logger.warning(f"Connection issue while reading: {e}")
                            break

                    if (
                        b"\n" in response_buffer
                    ):  # Check again after potential stderr read
                        break

                    await asyncio.sleep(
                        0.05
                    )  # Prevent tight loop if select has frequent small timeouts

                if hasattr(attached_socket, "_sock"):
                    attached_socket._sock.close()  # Close the underlying socket
                elif hasattr(attached_socket, "close"):
                    attached_socket.close()  # Close the stream object if it has a close method

                response_str = response_buffer.decode("utf-8", errors="replace").strip()
                if response_str:
                    logger.info(f"MCP Response from stdout: {response_str}")
                    try:
                        response_json = json.loads(response_str)
                        logger.info(f"Parsed MCP Response JSON: {response_json}")
                    except json.JSONDecodeError as je:
                        logger.error(f"Failed to parse MCP response as JSON: {je}")
                else:
                    logger.warning("No response received on stdout for MCP call.")

            except Exception as attach_e:
                logger.error(f"Error during MCP stdio call: {attach_e}", exc_info=True)
        else:
            logger.warning(
                f"Container {refetched_container.name} is not running (status: {refetched_container.status}). Cannot perform MCP call."
            )
            logger.info(
                f"Fetching logs for non-running container '{refetched_container.name}' (ID: {refetched_container.id}):"
            )
            try:
                logs = refetched_container.logs(stdout=True, stderr=True, stream=False)
                decoded_logs = logs.decode("utf-8", errors="replace")
                if decoded_logs:
                    logger.info("------ Container Logs Start ------")
                    print(decoded_logs.strip())
                    logger.info("------- Container Logs End -------")
                else:
                    logger.info("Container produced no logs.")
            except Exception as log_e:
                logger.error(
                    f"Error fetching logs for container {refetched_container.id}: {log_e}"
                )

    except docker.errors.DockerException as e:
        logger.error(f"A DockerException occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if container:
            try:
                # Fetch the container again to ensure we have the latest status
                c_to_remove = client.containers.get(container.id)
                logger.info(
                    f"Final status of container '{c_to_remove.name}' before cleanup: {c_to_remove.status}"
                )
                if c_to_remove.status == "running":
                    logger.info(f"Stopping running container '{c_to_remove.name}'...")
                    c_to_remove.stop(timeout=5)
                logger.info(f"Removing container '{c_to_remove.name}'...")
                c_to_remove.remove(force=True)
                logger.info(f"Container '{c_to_remove.name}' stopped and removed.")
            except docker.errors.NotFound:
                logger.info(
                    f"Container '{container_name}' already removed or not found for cleanup."
                )
            except Exception as e_clean:
                logger.error(
                    f"Error during cleanup of container '{container_name}': {e_clean}"
                )

        if temp_host_dir and os.path.exists(temp_host_dir):
            try:
                shutil.rmtree(temp_host_dir)
                logger.info(f"Removed temporary host directory: {temp_host_dir}")
            except Exception as e_rmdir:
                logger.error(
                    f"Error removing temporary host directory {temp_host_dir}: {e_rmdir}"
                )
        if client:
            client.close()
        logger.info("Test script finished.")


if __name__ == "__main__":
    asyncio.run(main())
