from typing import Any, Dict, List, Optional

from reward_kit.models import EvaluateResult, Message, MetricResult
from reward_kit.typed_interface import reward_function


@reward_function
def evaluate(
    messages: List[Message],
    ground_truth: str,
    final_filesystem_state: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
) -> EvaluateResult:
    """
    Pure evaluation function that compares actual vs expected filesystem state.
    This function is designed to be easily used by the Reward Kit framework for multiple rollouts.

    Args:
        messages: List of conversation messages from the rollout.
        ground_truth: Expected outcome description for this task.
        final_filesystem_state: The state of the filesystem at the end of the rollout,
                                 provided by the framework via the capture_function.
        task_description: The description of the task being evaluated.

    Returns:
        EvaluateResult containing the score and detailed metrics.
    """
    # Extract the model's final response
    completion = messages[-1].content if messages and messages[-1].content else ""

    # Use provided arguments directly
    final_state = final_filesystem_state if final_filesystem_state is not None else {}
    current_task_description = task_description if task_description is not None else ""

    # Parse ground truth expectation
    expected_state = parse_ground_truth(ground_truth)

    # Compare actual vs expected filesystem state
    filesystem_score = calculate_filesystem_match_score(final_state, expected_state)

    # Analyze the conversation for task completion indicators
    completion_score = analyze_task_completion(completion, current_task_description)

    # Combined score
    final_score = min(filesystem_score * completion_score, 1.0)

    return EvaluateResult(
        score=final_score,
        is_score_valid=True,
        reason=f"Filesystem match: {filesystem_score:.2f}, Task completion: {completion_score:.2f}",
        metrics={
            "filesystem_accuracy": MetricResult(
                score=filesystem_score,
                is_score_valid=True,
                reason="Comparison of expected vs actual filesystem state",
            ),
            "task_completion": MetricResult(
                score=completion_score,
                is_score_valid=True,
                reason="Analysis of completion indicators in model response",
            ),
        },
    )


def parse_ground_truth(ground_truth: str) -> Dict[str, Any]:
    """Parse ground truth string into expected filesystem state."""
    if "move file_to_move.txt to target_dir" in ground_truth:
        return {
            "expected_operations": ["move"],
            "source_files": ["file_to_move.txt"],
            "target_location": "target_dir/file_to_move.txt",
        }
    elif "copy all .txt files to backup_dir" in ground_truth:
        return {
            "expected_operations": ["copy"],
            "source_pattern": "*.txt",
            "target_location": "backup_dir/",
        }
    elif "create report.txt with specified content" in ground_truth:
        return {
            "expected_operations": ["create"],
            "target_file": "report.txt",
            "expected_content": "Task completed successfully",
        }
    return {"expected_operations": []}


def calculate_filesystem_match_score(actual_state: Dict, expected_state: Dict) -> float:
    """Calculate how well the actual filesystem state matches expectations."""
    if not expected_state.get("expected_operations"):
        return 1.0

    workspace_tree = actual_state.get("workspace_tree", {})
    operations = expected_state.get("expected_operations", [])

    if "move" in operations:
        return check_move_operation(workspace_tree, expected_state)
    elif "copy" in operations:
        return check_copy_operation(workspace_tree, expected_state)
    elif "create" in operations:
        return check_create_operation(workspace_tree, expected_state)

    return 0.0


def check_move_operation(workspace_tree: Dict, expected_state: Dict) -> float:
    """Check if file was moved correctly."""
    target_path = expected_state.get("target_location", "")
    path_parts = target_path.split("/")

    # Navigate to target location
    current = workspace_tree
    for part in path_parts[:-1]:
        if current.get("type") == "directory":
            current = current.get("children", {}).get(part, {})
        else:
            return 0.0

    # Check if file exists at target
    filename = path_parts[-1]
    if (
        current.get("type") == "directory"
        and filename in current.get("children", {})
        and current["children"][filename].get("type") == "file"
    ):
        return 1.0

    return 0.0


def check_copy_operation(workspace_tree: Dict, expected_state: Dict) -> float:
    """Check if files were copied correctly."""
    # Simplified check - look for files in target directory
    target_dir = expected_state.get("target_location", "").rstrip("/")

    if workspace_tree.get("type") == "directory":
        children = workspace_tree.get("children", {})
        if target_dir in children and children[target_dir].get("type") == "directory":
            target_children = children[target_dir].get("children", {})
            # Check if any .txt files exist in target
            txt_files = [f for f in target_children.keys() if f.endswith(".txt")]
            return 1.0 if txt_files else 0.0

    return 0.0


def check_create_operation(workspace_tree: Dict, expected_state: Dict) -> float:
    """Check if file was created with correct content."""
    target_file = expected_state.get("target_file", "")
    expected_content = expected_state.get("expected_content", "")

    if workspace_tree.get("type") == "directory":
        children = workspace_tree.get("children", {})
        if target_file in children and children[target_file].get("type") == "file":
            actual_content = children[target_file].get("content", "")
            return 1.0 if expected_content in actual_content else 0.5

    return 0.0


def analyze_task_completion(completion: str, task_description: str) -> float:
    """Analyze the model's response for task completion indicators."""
    if not completion:
        return 0.0

    completion_lower = completion.lower()

    # Positive indicators
    positive_indicators = [
        "successfully moved",
        "file has been moved",
        "completed the task",
        "operation successful",
        "task completed",
        "successfully copied",
        "file created",
        "successfully created",
    ]

    # Negative indicators
    negative_indicators = [
        "error",
        "failed",
        "cannot",
        "unable",
        "not found",
        "permission denied",
        "access denied",
    ]

    score = 0.5  # Base score

    # Boost for positive indicators
    if any(indicator in completion_lower for indicator in positive_indicators):
        score += 0.3

    # Penalty for negative indicators
    if any(indicator in completion_lower for indicator in negative_indicators):
        score -= 0.4

    # Bonus for detailed responses
    if len(completion) > 50:
        score += 0.2

    return max(0.0, min(1.0, score))
