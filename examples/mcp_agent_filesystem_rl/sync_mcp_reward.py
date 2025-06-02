"""
Synchronous MCP-based reward function for reward-kit CLI integration.

This provides a synchronous implementation that can be called directly
by the reward-kit evaluation pipeline without async/await complications.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from reward_kit.models import EvaluateResult, Message, MetricResult

logger = logging.getLogger(__name__)


class SyncMockMCPClient:
    """Synchronous mock MCP client for filesystem operations."""

    def __init__(self):
        self.sessions = {}
        self.next_session_id = 1

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock MCP tool calls with realistic filesystem simulation."""
        if tool_name == "initialize_session":
            session_id = f"session_{self.next_session_id}"
            self.next_session_id += 1

            # Get template path from first backend request
            template_details = args["backends"][0].get("template_details", "")
            initial_state = self._get_initial_state()

            self.sessions[session_id] = {
                "filesystem_state": initial_state,
                "operations_log": [],
            }

            return {
                "rk_session_id": session_id,
                "initialized_backends": [
                    {
                        "backend_name_ref": "filesystem_rl_example",
                        "instances": [{"instance_id": "instance_1"}],
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

            # Log operation
            self.sessions[session_id]["operations_log"].append(
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

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial filesystem state."""
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

    def _handle_filesystem_operation(
        self, session_id: str, operation: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle filesystem operations and update state."""
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
        # Get source content
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

        # Remove from source
        self._remove_file(fs_state, source)

        logger.info(f"Moved {source} to {destination}")
        return {"success": True, "message": f"Moved {source} to {destination}"}

    def _copy_file(
        self, fs_state: Dict, source: str, destination: str
    ) -> Dict[str, Any]:
        """Copy a file from source to destination."""
        source_item = self._navigate_to_path(fs_state, source)
        if isinstance(source_item, dict) and "error" in source_item:
            return source_item

        if source_item.get("type") != "file":
            return {"error": f"Source is not a file: {source}"}

        result = self._write_file(fs_state, destination, source_item.get("content", ""))
        if "success" in result:
            logger.info(f"Copied {source} to {destination}")
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

    def get_filesystem_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current filesystem state."""
        if session_id in self.sessions:
            return self.sessions[session_id]["filesystem_state"]
        return {}


class SyncMCPAgent:
    """Synchronous MCP agent for filesystem operations."""

    def __init__(self, client: SyncMockMCPClient, session_id: str):
        self.client = client
        self.session_id = session_id

    def execute_task(
        self, task_definition: Dict[str, Any]
    ) -> tuple[List[Message], Dict[str, Any], Optional[Dict[str, str]]]:
        """Execute a filesystem task and return messages, final state, and completion signal."""
        messages = []
        task_id = task_definition["id"]

        # Initial message
        user_query = task_definition.get(
            "user_instruction", task_definition.get("user_query", "")
        )
        messages.append(Message(role="user", content=user_query))

        # Execute task based on type
        if "move" in task_id:
            self._execute_move_task(messages)
        elif "copy" in task_id:
            self._execute_copy_task(messages)
        elif "create" in task_id:
            self._execute_create_task(messages)
        else:
            messages.append(Message(role="assistant", content="Unknown task type"))

        # Get final state
        final_state = self.client.get_filesystem_state(self.session_id)

        return messages, final_state

    def _execute_move_task(self, messages: List[Message]):
        """Execute file move task."""
        # List directories first
        list_result = self.client.call_tool(
            "call_backend_tool",
            {
                "rk_session_id": self.session_id,
                "backend_name_ref": "filesystem_rl_example",
                "instance_id": "instance_1",
                "tool_name": "list_directory",
                "tool_args": {"path": "/data"},
            },
        )
        messages.append(
            Message(
                role="assistant",
                content="I'll check the directory structure and then move the file.",
            )
        )

        # Move the file
        move_result = self.client.call_tool(
            "call_backend_tool",
            {
                "rk_session_id": self.session_id,
                "backend_name_ref": "filesystem_rl_example",
                "instance_id": "instance_1",
                "tool_name": "move_file",
                "tool_args": {
                    "source": "/data/source_dir/file_to_move.txt",
                    "destination": "/data/target_dir/file_to_move.txt",
                },
            },
        )

        if "success" in move_result:
            messages.append(
                Message(
                    role="assistant",
                    content="Successfully moved file_to_move.txt from source_dir to target_dir.",
                )
            )

            # Create completion signal
            self.client.call_tool(
                "call_backend_tool",
                {
                    "rk_session_id": self.session_id,
                    "backend_name_ref": "filesystem_rl_example",
                    "instance_id": "instance_1",
                    "tool_name": "write_file",
                    "tool_args": {
                        "path": "/data/.task_status.json",
                        "content": json.dumps(
                            {
                                "status": "completed",
                                "message": "Successfully moved file_to_move.txt from source_dir to target_dir",
                            }
                        ),
                    },
                },
            )
        else:
            messages.append(
                Message(
                    role="assistant",
                    content=f"Failed to move file: {move_result.get('error', 'Unknown error')}",
                )
            )

    def _execute_copy_task(self, messages: List[Message]):
        """Execute file copy task."""
        messages.append(
            Message(
                role="assistant",
                content="I'll copy all .txt files from source_dir to backup_dir.",
            )
        )

        # List source directory
        source_result = self.client.call_tool(
            "call_backend_tool",
            {
                "rk_session_id": self.session_id,
                "backend_name_ref": "filesystem_rl_example",
                "instance_id": "instance_1",
                "tool_name": "list_directory",
                "tool_args": {"path": "/data/source_dir"},
            },
        )

        # Copy each .txt file
        success_count = 0
        for entry in source_result.get("entries", []):
            if entry["name"].endswith(".txt"):
                copy_result = self.client.call_tool(
                    "call_backend_tool",
                    {
                        "rk_session_id": self.session_id,
                        "backend_name_ref": "filesystem_rl_example",
                        "instance_id": "instance_1",
                        "tool_name": "copy_file",
                        "tool_args": {
                            "source": f"/data/source_dir/{entry['name']}",
                            "destination": f"/data/backup_dir/{entry['name']}",
                        },
                    },
                )
                if "success" in copy_result:
                    success_count += 1

        if success_count > 0:
            messages.append(
                Message(
                    role="assistant",
                    content=f"Successfully copied {success_count} .txt files to backup_dir.",
                )
            )

            # Create completion signal
            self.client.call_tool(
                "call_backend_tool",
                {
                    "rk_session_id": self.session_id,
                    "backend_name_ref": "filesystem_rl_example",
                    "instance_id": "instance_1",
                    "tool_name": "write_file",
                    "tool_args": {
                        "path": "/data/.task_status.json",
                        "content": json.dumps(
                            {
                                "status": "completed",
                                "message": f"Successfully copied {success_count} .txt files to backup_dir",
                            }
                        ),
                    },
                },
            )
        else:
            messages.append(Message(role="assistant", content="Failed to copy files."))

    def _execute_create_task(self, messages: List[Message]):
        """Execute file creation task."""
        messages.append(
            Message(
                role="assistant",
                content="I'll create the report.txt file with the specified content.",
            )
        )

        # Create the file
        create_result = self.client.call_tool(
            "call_backend_tool",
            {
                "rk_session_id": self.session_id,
                "backend_name_ref": "filesystem_rl_example",
                "instance_id": "instance_1",
                "tool_name": "write_file",
                "tool_args": {
                    "path": "/data/report.txt",
                    "content": "Task completed successfully",
                },
            },
        )

        if "success" in create_result:
            messages.append(
                Message(
                    role="assistant",
                    content="Successfully created report.txt with the content 'Task completed successfully'.",
                )
            )

            # Create completion signal
            self.client.call_tool(
                "call_backend_tool",
                {
                    "rk_session_id": self.session_id,
                    "backend_name_ref": "filesystem_rl_example",
                    "instance_id": "instance_1",
                    "tool_name": "write_file",
                    "tool_args": {
                        "path": "/data/.task_status.json",
                        "content": json.dumps(
                            {
                                "status": "completed",
                                "message": "Successfully created report.txt with the specified content",
                            }
                        ),
                    },
                },
            )
        else:
            messages.append(
                Message(
                    role="assistant",
                    content=f"Failed to create file: {create_result.get('error', 'Unknown error')}",
                )
            )


def compare_filesystem_trees(actual: Dict, expected: Dict, path: str = "/") -> float:
    """Recursively compare filesystem trees."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return 1.0 if actual == expected else 0.0

    # Compare type
    actual_type = actual.get("type", "unknown")
    expected_type = expected.get("type", "unknown")

    if actual_type != expected_type:
        return 0.0

    if actual_type == "file":
        # Compare file content
        actual_content = actual.get("content", "")
        expected_content = expected.get("content", "")
        return 1.0 if actual_content == expected_content else 0.5

    elif actual_type == "directory":
        # Compare directory contents
        actual_contents = actual.get("contents", {})
        expected_contents = expected.get("contents", {})

        if not expected_contents:  # No specific expectations
            return 1.0

        scores = []
        for name, expected_item in expected_contents.items():
            if name in actual_contents:
                item_score = compare_filesystem_trees(
                    actual_contents[name], expected_item, f"{path.rstrip('/')}/{name}"
                )
                scores.append(item_score)
            else:
                scores.append(0.0)  # Missing expected item

        return sum(scores) / len(scores) if scores else 1.0

    return 1.0


def analyze_agent_completion_behavior(
    messages: List[Message], filesystem_score: float
) -> tuple[float, str]:
    """Analyze the agent's completion behavior from message trajectory."""
    # Look for completion indicators in the agent's messages
    agent_messages = [msg for msg in messages if msg.role == "assistant"]

    if not agent_messages:
        return 0.1, "No agent messages found"

    # Check the last few messages for completion indicators
    completion_indicators = []
    for msg in agent_messages[-3:]:  # Check last 3 agent messages
        content = msg.content.lower()

        # Look for explicit completion claims
        if any(
            phrase in content
            for phrase in [
                "task completed",
                "successfully",
                "done",
                "finished",
                "moved",
                "copied",
                "created",
                "organized",
            ]
        ):
            completion_indicators.append("claimed_success")

        # Look for task status file creation
        if ".task_status.json" in content:
            completion_indicators.append("created_status_file")

        # Look for uncertainty or problems
        if any(
            phrase in content
            for phrase in [
                "error",
                "failed",
                "couldn't",
                "unable",
                "not sure",
                "problem",
            ]
        ):
            completion_indicators.append("indicated_problems")

    # Analyze completion behavior
    if "claimed_success" in completion_indicators:
        if filesystem_score >= 0.8:
            return 1.0, "Agent correctly identified successful completion"
        elif filesystem_score >= 0.5:
            return 0.7, "Agent was optimistic but task was partially successful"
        else:
            return 0.3, "Agent incorrectly claimed completion of failed task"

    elif "indicated_problems" in completion_indicators:
        if filesystem_score < 0.5:
            return 1.0, "Agent correctly identified task problems"
        else:
            return 0.2, "Agent incorrectly indicated problems with successful task"

    elif "created_status_file" in completion_indicators:
        # Agent followed protocol but didn't explicitly claim success/failure
        return 0.6, "Agent followed completion protocol but was unclear about success"

    else:
        return 0.2, "Agent did not clearly indicate task completion status"


def sync_mcp_filesystem_reward(
    messages: List[Any], ground_truth: str = "", **sample_data
) -> EvaluateResult:
    """
    Synchronous MCP-based reward function.

    This function performs actual filesystem operations via MCP and evaluates
    the results against expected filesystem states.
    """
    try:
        # Extract task definition
        task_definition = {
            "id": sample_data.get("id", "unknown"),
            "user_query": sample_data.get("user_query", ""),
            "user_instruction": sample_data.get(
                "user_instruction", sample_data.get("user_query", "")
            ),
            "expected_final_state": sample_data.get("expected_final_state"),
            "ground_truth_for_eval": ground_truth,
        }

        # Create MCP client and initialize session
        client = SyncMockMCPClient()

        # Initialize session
        init_result = client.call_tool(
            "initialize_session",
            {
                "backends": [
                    {
                        "backend_name_ref": "filesystem_rl_example",
                        "num_instances": 1,
                        "template_details": "mock_template",
                    }
                ]
            },
        )

        session_id = init_result["rk_session_id"]

        # Execute task with agent
        agent = SyncMCPAgent(client, session_id)
        mcp_messages, final_filesystem_state = agent.execute_task(task_definition)

        # Get expected state
        expected_state = task_definition.get("expected_final_state")
        if not expected_state:
            raise ValueError("No expected filesystem state provided")

        # Compare filesystem states
        filesystem_score = compare_filesystem_trees(
            final_filesystem_state, expected_state
        )

        # Analyze agent completion behavior from messages
        signal_score, signal_analysis = analyze_agent_completion_behavior(
            mcp_messages, filesystem_score
        )

        # Calculate final score
        final_score = filesystem_score * 0.8 + signal_score * 0.2

        reason_str = f"Filesystem accuracy: {filesystem_score:.2f}, Agent signal: {signal_score:.2f}"

        logger.info(
            f"MCP evaluation for {task_definition['id']}: {final_score:.2f} - {reason_str}"
        )

        # Cleanup
        client.call_tool("cleanup_session", {"rk_session_id": session_id})

        return EvaluateResult(
            score=final_score,
            is_score_valid=True,
            reason=reason_str,
            metrics={
                "filesystem_accuracy": MetricResult(
                    score=filesystem_score,
                    is_score_valid=True,
                    reason="Direct comparison of actual vs expected filesystem state",
                ),
                "agent_completion_behavior": MetricResult(
                    score=signal_score,
                    is_score_valid=True,
                    reason=signal_analysis,
                ),
            },
        )

    except Exception as e:
        logger.error(f"Error in sync MCP reward function: {e}")
        return EvaluateResult(
            score=0.0, is_score_valid=False, reason=f"Error: {str(e)}", metrics={}
        )
