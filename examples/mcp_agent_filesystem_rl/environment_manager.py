import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def setup_environment(task_id: str, template_path: str = None) -> Dict[str, Any]:
    """
    Initialize a clean environment for the task.

    Args:
        task_id: Unique identifier for this task instance
        template_path: Path to template directory to copy

    Returns:
        Dict containing environment setup information
    """
    if template_path is None:
        template_path = str(Path(__file__).parent / "templates" / "workspace")

    logger.info(f"Setting up environment for task {task_id}")

    return {"template_path": template_path, "initialized": True, "task_id": task_id}


def cleanup_environment(
    task_id: str, container_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Clean up resources after task execution.

    Args:
        task_id: Unique identifier for this task instance
        container_info: Information about container resources

    Returns:
        Dict containing cleanup status
    """
    logger.info(f"Cleanup initiated for task {task_id}")

    # Cleanup is handled by the framework (Docker container removal)
    # This function can perform any additional cleanup logic

    return {"cleanup_completed": True, "task_id": task_id, "status": "success"}


def capture_filesystem_state(container_workspace: str = "/workspace") -> Dict[str, Any]:
    """
    Capture the current state of the filesystem for comparison.

    Args:
        container_workspace: Path to workspace inside container

    Returns:
        Dict containing filesystem state information
    """

    def get_file_tree(path: str) -> Dict[str, Any]:
        """Recursively build a file tree structure."""
        try:
            if os.path.isfile(path):
                with open(path, "r", errors="ignore") as f:
                    content = f.read()
                return {"type": "file", "content": content}
            elif os.path.isdir(path):
                result = {"type": "directory", "children": {}}
                for item in sorted(os.listdir(path)):
                    item_path = os.path.join(path, item)
                    result["children"][item] = get_file_tree(item_path)
                return result
        except (PermissionError, FileNotFoundError):
            return {"type": "error", "message": "Access denied or file not found"}

        return {"type": "unknown"}

    return {
        "workspace_tree": get_file_tree(container_workspace),
        "captured_at": "end_of_execution",
    }
