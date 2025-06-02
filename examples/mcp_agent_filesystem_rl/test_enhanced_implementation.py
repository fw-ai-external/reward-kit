#!/usr/bin/env python3
"""
Test script for the enhanced MCP Agent Filesystem RL implementation.

This demonstrates the key features implemented according to the plan:
1. Enhanced EnvironmentManager with real filesystem state verification
2. Multi-turn dialogue capabilities (both LLM and scripted modes)
3. Updated reward function for structured filesystem comparison
4. New dataset format with expected filesystem states
5. Task completion signaling protocol
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the reward-kit root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.mcp_agent_filesystem_rl.environment_manager import EnvironmentManager
from examples.mcp_agent_filesystem_rl.main import evaluate
from examples.mcp_agent_filesystem_rl.user_simulator import (
    UserSimulatorLLM,
    UserSimulatorScripted,
    is_dialogue_complete,
)
from reward_kit.models import Message

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockMCPClient:
    """Mock MCP client for testing the enhanced implementation."""

    def __init__(self):
        self.sessions = {}
        self.next_session_id = 1

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock MCP tool calls with realistic filesystem simulation."""
        logger.debug(f"MockMCPClient: {tool_name} called with {args}")

        if tool_name == "initialize_session":
            session_id = f"mock_session_{self.next_session_id}"
            self.next_session_id += 1

            # Initialize with scenario-specific state
            template_path = args["backends"][0].get("template_data_path_host", "")
            initial_state = self._get_initial_state_for_template(template_path)

            self.sessions[session_id] = {
                "filesystem_state": initial_state,
                "operations_performed": [],
            }

            return {
                "rk_session_id": session_id,
                "initialized_backends": [
                    {
                        "backend_name_ref": "filesystem_rl_example",
                        "instances": [{"instance_id": "mock_instance_1"}],
                    }
                ],
            }

        elif tool_name == "call_backend_tool":
            session_id = args["rk_session_id"]
            backend_tool = args["tool_name"]
            tool_args = args.get("tool_args", {})

            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            result = self._handle_filesystem_operation(
                session_id, backend_tool, tool_args
            )

            # Log operation for tracking
            self.sessions[session_id]["operations_performed"].append(
                {"tool": backend_tool, "args": tool_args, "result": result}
            )

            return result

        elif tool_name == "cleanup_session":
            session_id = args["rk_session_id"]
            if session_id in self.sessions:
                del self.sessions[session_id]
            return {"status": "cleaned", "rk_session_id": session_id}

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _get_initial_state_for_template(self, template_path: str) -> Dict[str, Any]:
        """Get initial filesystem state based on template path."""
        # Determine scenario from template path
        if "scenario_move_001" in template_path:
            return {
                "type": "directory",
                "name": "data",
                "contents": {
                    "source_dir": {
                        "type": "directory",
                        "name": "source_dir",
                        "contents": {
                            "file_to_move.txt": {
                                "type": "file",
                                "content": "This file should be moved.",
                            },
                            "sample.txt": {
                                "type": "file",
                                "content": "This is a sample file.",
                            },
                        },
                    },
                    "target_dir": {
                        "type": "directory",
                        "name": "target_dir",
                        "contents": {},
                    },
                    "backup_dir": {
                        "type": "directory",
                        "name": "backup_dir",
                        "contents": {},
                    },
                },
            }
        elif "scenario_copy_001" in template_path:
            return {
                "type": "directory",
                "name": "data",
                "contents": {
                    "source_dir": {
                        "type": "directory",
                        "name": "source_dir",
                        "contents": {
                            "file_to_move.txt": {
                                "type": "file",
                                "content": "This file should be moved.",
                            },
                            "sample.txt": {
                                "type": "file",
                                "content": "This is a sample file.",
                            },
                        },
                    },
                    "target_dir": {
                        "type": "directory",
                        "name": "target_dir",
                        "contents": {},
                    },
                    "backup_dir": {
                        "type": "directory",
                        "name": "backup_dir",
                        "contents": {},
                    },
                },
            }
        else:
            # Default state
            return {"type": "directory", "name": "data", "contents": {}}

    def _handle_filesystem_operation(
        self, session_id: str, operation: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle filesystem operations with realistic behavior."""
        session = self.sessions[session_id]
        fs_state = session["filesystem_state"]

        if operation == "list_directory":
            path = args.get("path", "/data")
            return self._list_directory(fs_state, path)

        elif operation == "read_file":
            path = args.get("path", "")
            return self._read_file(fs_state, path)

        elif operation == "move_file":
            source = args.get("source", "")
            destination = args.get("destination", "")
            return self._move_file(fs_state, source, destination)

        elif operation == "copy_file":
            source = args.get("source", "")
            destination = args.get("destination", "")
            return self._copy_file(fs_state, source, destination)

        elif operation == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            return self._write_file(fs_state, path, content)

        elif operation == "create_directory":
            path = args.get("path", "")
            return self._create_directory(fs_state, path)

        else:
            return {"error": f"Unknown operation: {operation}"}

    def _list_directory(self, fs_state: Dict, path: str) -> Dict[str, Any]:
        """List directory contents."""
        current = self._navigate_to_path(fs_state, path)
        if isinstance(current, dict) and "error" in current:
            return current

        if current.get("type") != "directory":
            return {"error": f"Not a directory: {path}"}

        entries = []
        for name, item in current.get("contents", {}).items():
            entries.append({"name": name, "type": item.get("type", "unknown")})

        return {"entries": entries}

    def _read_file(self, fs_state: Dict, path: str) -> Dict[str, Any]:
        """Read file content."""
        current = self._navigate_to_path(fs_state, path)
        if isinstance(current, dict) and "error" in current:
            return current

        if current.get("type") != "file":
            return {"error": f"Not a file: {path}"}

        return {"content": current.get("content", "")}

    def _move_file(
        self, fs_state: Dict, source: str, destination: str
    ) -> Dict[str, Any]:
        """Move a file from source to destination."""
        # Read source content
        source_item = self._navigate_to_path(fs_state, source)
        if isinstance(source_item, dict) and "error" in source_item:
            return source_item

        if source_item.get("type") != "file":
            return {"error": f"Source is not a file: {source}"}

        # Write to destination
        write_result = self._write_file(
            fs_state, destination, source_item.get("content", "")
        )
        if "error" in write_result:
            return write_result

        # Remove from source - this actually modifies the filesystem state
        self._remove_file(fs_state, source)

        logger.info(f"Mock filesystem: Moved {source} to {destination}")
        return {"success": True, "message": f"Moved {source} to {destination}"}

    def _copy_file(
        self, fs_state: Dict, source: str, destination: str
    ) -> Dict[str, Any]:
        """Copy a file from source to destination."""
        # Read source content
        source_item = self._navigate_to_path(fs_state, source)
        if isinstance(source_item, dict) and "error" in source_item:
            return source_item

        if source_item.get("type") != "file":
            return {"error": f"Source is not a file: {source}"}

        # Write to destination
        result = self._write_file(fs_state, destination, source_item.get("content", ""))
        if "success" in result:
            logger.info(f"Mock filesystem: Copied {source} to {destination}")
        return result

    def _write_file(self, fs_state: Dict, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        path_parts = [p for p in path.split("/") if p and p != "data"]

        if not path_parts:
            return {"error": "Invalid path"}

        # Navigate to parent directory
        current = fs_state
        for part in path_parts[:-1]:
            if current.get("type") == "directory" and part in current.get(
                "contents", {}
            ):
                current = current["contents"][part]
            else:
                return {"error": f"Parent directory not found: {path}"}

        filename = path_parts[-1]
        if current.get("type") == "directory":
            current.setdefault("contents", {})[filename] = {
                "type": "file",
                "content": content,
            }
            return {"success": True, "message": f"File written: {path}"}
        else:
            return {"error": f"Parent is not a directory: {path}"}

    def _create_directory(self, fs_state: Dict, path: str) -> Dict[str, Any]:
        """Create a directory."""
        path_parts = [p for p in path.split("/") if p and p != "data"]

        if not path_parts:
            return {"error": "Invalid path"}

        # Navigate to parent directory
        current = fs_state
        for part in path_parts[:-1]:
            if current.get("type") == "directory" and part in current.get(
                "contents", {}
            ):
                current = current["contents"][part]
            else:
                return {"error": f"Parent directory not found: {path}"}

        dirname = path_parts[-1]
        if current.get("type") == "directory":
            current.setdefault("contents", {})[dirname] = {
                "type": "directory",
                "name": dirname,
                "contents": {},
            }
            return {"success": True, "message": f"Directory created: {path}"}
        else:
            return {"error": f"Parent is not a directory: {path}"}

    def _remove_file(self, fs_state: Dict, path: str):
        """Remove a file from the filesystem."""
        path_parts = [p for p in path.split("/") if p and p != "data"]

        if not path_parts:
            return

        # Navigate to parent directory
        current = fs_state
        for part in path_parts[:-1]:
            if current.get("type") == "directory" and part in current.get(
                "contents", {}
            ):
                current = current["contents"][part]
            else:
                return

        filename = path_parts[-1]
        if current.get("type") == "directory" and filename in current.get(
            "contents", {}
        ):
            del current["contents"][filename]

    def _navigate_to_path(self, fs_state: Dict, path: str) -> Dict[str, Any]:
        """Navigate to a specific path in the filesystem."""
        current = fs_state
        path_parts = [p for p in path.split("/") if p and p != "data"]

        for part in path_parts:
            if current.get("type") == "directory" and part in current.get(
                "contents", {}
            ):
                current = current["contents"][part]
            else:
                return {"error": f"Path not found: {path}"}

        return current


async def test_enhanced_features():
    """Test all enhanced features of the implementation."""
    logger.info("=== Testing Enhanced MCP Agent Filesystem RL Implementation ===")

    # Load test tasks from enhanced dataset
    dataset_path = Path(__file__).parent / "dataset.jsonl"
    tasks = []
    with open(dataset_path, "r") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(tasks)} test tasks")

    results = []

    for task in tasks:
        logger.info(f"\n--- Testing Task: {task['id']} ---")
        result = await test_single_task(task)
        results.append(result)

        logger.info(f"Task {task['id']} completed with score: {result['score']:.2f}")

    # Print summary
    logger.info(f"\n=== SUMMARY ===")
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0.0

    logger.info(f"Total tasks: {len(results)}")
    logger.info(f"Average score: {avg_score:.2f}")

    for result in results:
        logger.info(
            f"  {result['task_id']}: {result['score']:.2f} ({result['evaluation_mode']})"
        )

    return avg_score >= 0.8  # Success threshold


async def test_single_task(task_definition: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single task with the enhanced implementation."""
    task_id = task_definition["id"]

    # Create mock MCP client
    mock_client = MockMCPClient()

    # Create environment manager
    base_template_path = str(Path(__file__).parent / "templates" / "workspace")
    env_manager = EnvironmentManager(mock_client, base_template_path)

    try:
        # Set up environment
        await env_manager.setup_environment(
            task_id=task_id,
            template_subdir=task_definition["initial_template_subdir"],
            dialogue_mode=task_definition["dialogue_mode"],
            user_instruction=task_definition.get("user_instruction", ""),
            scripted_user_turns=task_definition.get("scripted_user_turns", []),
            user_llm_config={
                "model": "mock-model",
                "temperature": 0.7,
                "max_tokens": 150,
            },
        )

        logger.info(f"Environment set up for {task_id}")

        # Run dialogue simulation
        messages = await simulate_agent_task(env_manager, task_definition)

        # Capture final filesystem state
        final_filesystem_state = await env_manager.get_filesystem_state()
        logger.info(
            f"Captured final filesystem state with {len(final_filesystem_state.get('contents', {}))} top-level items"
        )

        # Check for completion signal
        completion_signal = await env_manager.check_task_completion_signal()
        logger.info(f"Agent completion signal: {completion_signal}")

        # Convert messages to Message objects
        message_objects = [
            Message(role=msg["role"], content=msg["content"]) for msg in messages
        ]

        # Evaluate with enhanced reward function
        evaluation_result = evaluate(
            messages=message_objects,
            ground_truth=task_definition["ground_truth_for_eval"],
            final_filesystem_state=final_filesystem_state,
            task_description=task_definition.get("user_instruction", ""),
            expected_final_state=task_definition["expected_final_state"],
            agent_completion_signal=completion_signal,
            task_definition=task_definition,
        )

        # Extract evaluation mode from reason
        evaluation_mode = "unknown"
        if "Mode:" in evaluation_result.reason:
            evaluation_mode = (
                evaluation_result.reason.split("Mode:")[1].split(",")[0].strip()
            )

        logger.info(
            f"Evaluation: {evaluation_result.score:.2f} - {evaluation_result.reason}"
        )
        for metric_name, metric_result in evaluation_result.metrics.items():
            logger.info(f"  {metric_name}: {metric_result.score:.2f}")

        return {
            "task_id": task_id,
            "score": evaluation_result.score,
            "reason": evaluation_result.reason,
            "metrics": {
                name: result.score for name, result in evaluation_result.metrics.items()
            },
            "evaluation_mode": evaluation_mode,
            "messages_count": len(messages),
            "filesystem_state_captured": bool(final_filesystem_state),
            "completion_signal_present": bool(completion_signal),
        }

    except Exception as e:
        logger.error(f"Error testing task {task_id}: {e}")
        return {
            "task_id": task_id,
            "score": 0.0,
            "reason": f"Error: {str(e)}",
            "metrics": {},
            "evaluation_mode": "error",
            "messages_count": 0,
            "filesystem_state_captured": False,
            "completion_signal_present": False,
        }

    finally:
        # Clean up
        await env_manager.cleanup_environment()


async def simulate_agent_task(
    env_manager: EnvironmentManager, task_definition: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Simulate an intelligent agent performing the task."""
    messages = []
    task_id = task_definition["id"]

    # Get initial user query
    initial_query = await env_manager.get_initial_user_query()
    logger.info(f"Initial user query: {initial_query}")
    messages.append({"role": "user", "content": initial_query})

    # Simulate agent behavior based on task type
    max_turns = 8
    for turn in range(max_turns):
        agent_response = await simulate_agent_response(
            env_manager, task_definition, messages, turn
        )

        if not agent_response:
            break

        logger.info(f"Agent response {turn + 1}: {agent_response}")
        messages.append({"role": "assistant", "content": agent_response})

        # Check if we should continue the dialogue
        if task_definition["dialogue_mode"] == "scripted_user":
            try:
                user_response = await env_manager.get_next_user_response(agent_response)
                if is_dialogue_complete(user_response):
                    logger.info(f"User response: {user_response}")
                    messages.append({"role": "user", "content": user_response})
                    break
                elif user_response and user_response != "###SCRIPT_END###":
                    logger.info(f"User response: {user_response}")
                    messages.append({"role": "user", "content": user_response})
                else:
                    break
            except Exception as e:
                logger.warning(f"Error getting user response: {e}")
                break
        else:
            # For LLM user mode, simulate completion after successful task execution
            if turn >= 3 and "successfully" in agent_response.lower():
                user_response = "Thank you, that looks correct. ###TASK_SATISFIED###"
                logger.info(f"User response: {user_response}")
                messages.append({"role": "user", "content": user_response})
                break

    return messages


async def simulate_agent_response(
    env_manager: EnvironmentManager,
    task_definition: Dict[str, Any],
    messages: List[Dict],
    turn: int,
) -> str:
    """Simulate intelligent agent responses based on the task."""
    task_id = task_definition["id"]

    try:
        if "move" in task_id:
            return await simulate_move_task(env_manager, turn)
        elif "copy" in task_id:
            return await simulate_copy_task(env_manager, turn)
        elif "create" in task_id:
            return await simulate_create_task(env_manager, turn)
        else:
            return f"I'm working on this task (turn {turn + 1})"
    except Exception as e:
        logger.error(f"Error in agent simulation: {e}")
        return f"I encountered an error while working on the task: {str(e)}"


async def simulate_move_task(env_manager: EnvironmentManager, turn: int) -> str:
    """Simulate agent performing a file move task."""
    if turn == 0:
        # First, check the directory structure
        list_result = await env_manager.execute_mcp_action(
            "list_directory", {"path": "/data"}
        )
        return "I'll help you move the file. Let me first check the current directory structure."

    elif turn == 1:
        # Check source directory
        source_result = await env_manager.execute_mcp_action(
            "list_directory", {"path": "/data/source_dir"}
        )
        return "I can see the source_dir directory. Let me check what files are available to move."

    elif turn == 2:
        # Perform the move operation
        move_result = await env_manager.execute_mcp_action(
            "move_file",
            {
                "source": "/data/source_dir/file_to_move.txt",
                "destination": "/data/target_dir/file_to_move.txt",
            },
        )
        return "I'm moving the file_to_move.txt from source_dir to target_dir now."

    elif turn == 3:
        # Verify the move and create completion signal
        verify_result = await env_manager.execute_mcp_action(
            "list_directory", {"path": "/data/target_dir"}
        )

        # Create completion signal
        await env_manager.execute_mcp_action(
            "write_file",
            {
                "path": "/data/.task_status.json",
                "content": json.dumps(
                    {
                        "status": "completed",
                        "message": "Successfully moved file_to_move.txt from source_dir to target_dir",
                    }
                ),
            },
        )

        return "Task completed successfully! I've moved file_to_move.txt from source_dir to target_dir. The file should now be in the target directory."

    else:
        return "The file move operation has been completed."


async def simulate_copy_task(env_manager: EnvironmentManager, turn: int) -> str:
    """Simulate agent performing a file copy task."""
    if turn == 0:
        return "I'll help you copy the .txt files. Let me check the source directory first."

    elif turn == 1:
        # Check source directory
        source_result = await env_manager.execute_mcp_action(
            "list_directory", {"path": "/data/source_dir"}
        )
        return (
            "I can see the files in source_dir. I'll copy all .txt files to backup_dir."
        )

    elif turn == 2:
        # Copy files
        copy1_result = await env_manager.execute_mcp_action(
            "copy_file",
            {
                "source": "/data/source_dir/file_to_move.txt",
                "destination": "/data/backup_dir/file_to_move.txt",
            },
        )
        copy2_result = await env_manager.execute_mcp_action(
            "copy_file",
            {
                "source": "/data/source_dir/sample.txt",
                "destination": "/data/backup_dir/sample.txt",
            },
        )
        logger.info(f"Copy results: {copy1_result}, {copy2_result}")
        return "I'm copying the .txt files from source_dir to backup_dir."

    elif turn == 3:
        # Create completion signal
        await env_manager.execute_mcp_action(
            "write_file",
            {
                "path": "/data/.task_status.json",
                "content": json.dumps(
                    {
                        "status": "completed",
                        "message": "Successfully copied all .txt files to backup_dir",
                    }
                ),
            },
        )
        return (
            "All .txt files have been successfully copied to the backup_dir directory."
        )

    else:
        return "The copy operation has been completed."


async def simulate_create_task(env_manager: EnvironmentManager, turn: int) -> str:
    """Simulate agent performing a file creation task."""
    if turn == 0:
        return "I'll create the report.txt file for you with the specified content."

    elif turn == 1:
        # Create the file
        create_result = await env_manager.execute_mcp_action(
            "write_file",
            {"path": "/data/report.txt", "content": "Task completed successfully"},
        )
        return "I'm creating the report.txt file with the content 'Task completed successfully'."

    elif turn == 2:
        # Create completion signal
        await env_manager.execute_mcp_action(
            "write_file",
            {
                "path": "/data/.task_status.json",
                "content": json.dumps(
                    {
                        "status": "completed",
                        "message": "Successfully created report.txt with the specified content",
                    }
                ),
            },
        )
        return "The report.txt file has been successfully created in the main directory with the content 'Task completed successfully'."

    else:
        return "The file creation task has been completed."


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_features())
    if success:
        print("\n🎉 All enhanced features are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
