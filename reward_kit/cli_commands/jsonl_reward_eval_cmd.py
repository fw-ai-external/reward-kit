"""
CLI command for re-evaluating reward functions from JSONL trajectory files.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List

from reward_kit.models import EvaluateResult, Message, StepOutput
from reward_kit.utils.module_loader import load_function


def jsonl_reward_eval_command(args):
    """
    Re-evaluate reward functions using saved JSONL trajectory files.

    Args:
        args: Command line arguments containing:
            - jsonl_file: Path to JSONL file containing trajectory data
            - reward_module: Python module path for the reward function
            - output_file: Optional output file for re-evaluated results
    """
    logger = logging.getLogger("jsonl_reward_eval")
    logger.info("Starting JSONL reward re-evaluation command.")

    if not args.jsonl_file:
        logger.error("Error: --jsonl-file (path to trajectory JSONL file) is required.")
        return 1

    if not args.reward_module:
        logger.error("Error: --reward-module (Python module path) is required.")
        return 1

    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        return 1

    try:
        # Load the reward function
        logger.info(f"Loading reward function from module: {args.reward_module}")
        reward_function = load_function(args.reward_module)

        # Read and parse JSONL file
        logger.info(f"Reading trajectory data from: {jsonl_path}")
        trajectories = []
        summary_data = None

        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "summary":
                        summary_data = data
                    elif data.get("type") == "individual_result":
                        trajectories.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

        if not trajectories:
            logger.error("No trajectory data found in JSONL file")
            return 1

        logger.info(f"Found {len(trajectories)} trajectories to re-evaluate")

        # Re-evaluate each trajectory
        re_evaluated_results = []
        new_scores = []

        for i, trajectory in enumerate(trajectories):
            logger.info(f"Re-evaluating trajectory {i+1}/{len(trajectories)}")

            try:
                # Reconstruct the evaluation context
                eval_context = _reconstruct_eval_context(trajectory)

                # Call the reward function
                result = reward_function(**eval_context)

                # Convert EvaluateResult to dict if needed
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    result_dict = result

                # Store the re-evaluated result
                re_evaluated_result = {
                    "type": "re_evaluated_result",
                    "original_rollout_index": trajectory.get("rollout_index", i),
                    "original_score": trajectory.get("score", 0.0),
                    "new_score": result_dict.get("score", 0.0),
                    "new_reason": result_dict.get("reason", ""),
                    "new_metrics": result_dict.get("metrics", {}),
                    "reward_module": args.reward_module,
                    "original_trajectory": trajectory,
                }

                re_evaluated_results.append(re_evaluated_result)
                new_scores.append(result_dict.get("score", 0.0))

                logger.info(
                    f"Trajectory {i+1}: Original score: {trajectory.get('score', 0.0):.3f}, New score: {result_dict.get('score', 0.0):.3f}"
                )

            except Exception as e:
                logger.error(f"Error re-evaluating trajectory {i+1}: {e}")
                continue

        # Calculate aggregate statistics
        if new_scores:
            avg_original = sum(t.get("score", 0.0) for t in trajectories) / len(
                trajectories
            )
            avg_new = sum(new_scores) / len(new_scores)
            std_dev_new = statistics.stdev(new_scores) if len(new_scores) > 1 else 0.0

            aggregate_stats = {
                "type": "re_evaluation_summary",
                "reward_module": args.reward_module,
                "num_trajectories": len(trajectories),
                "original_avg_score": avg_original,
                "new_avg_score": avg_new,
                "new_std_dev": std_dev_new,
                "new_min_score": min(new_scores),
                "new_max_score": max(new_scores),
                "score_improvement": avg_new - avg_original,
            }

        # Output results
        output_file = args.output_file or f"re_evaluated_{jsonl_path.stem}.jsonl"
        output_path = Path(output_file)

        logger.info(f"Writing re-evaluated results to: {output_path}")
        with open(output_path, "w") as f:
            # Write aggregate statistics
            f.write(json.dumps(aggregate_stats) + "\n")

            # Write individual re-evaluated results
            for result in re_evaluated_results:
                f.write(json.dumps(result) + "\n")

        # Log summary statistics
        logger.info("=" * 60)
        logger.info("RE-EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Reward Module: {args.reward_module}")
        logger.info(f"Trajectories Re-evaluated: {len(trajectories)}")
        logger.info(f"Original Average Score: {avg_original:.4f}")
        logger.info(f"New Average Score: {avg_new:.4f}")
        logger.info(f"Score Improvement: {avg_new - avg_original:+.4f}")
        logger.info(f"New Standard Deviation: {std_dev_new:.4f}")
        logger.info(f"New Score Range: {min(new_scores):.4f} - {max(new_scores):.4f}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during JSONL reward re-evaluation: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


def _reconstruct_eval_context(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct the evaluation context from trajectory data.

    This function extracts the necessary information from a trajectory
    to call a reward function in the same way it was originally called.
    """
    # Check if we have the full reward function inputs stored
    reward_function_inputs = trajectory.get("reward_function_inputs")

    if reward_function_inputs:
        # Use the actual inputs that were passed to the reward function
        eval_context: Dict[str, Any] = {}

        # Convert message dicts back to Message objects if needed
        messages = reward_function_inputs.get("messages", [])
        if messages and isinstance(messages[0], dict):
            # Convert dict messages back to Message objects
            message_objects = []
            for msg_dict in messages:
                message_objects.append(Message(**msg_dict))
            eval_context["messages"] = message_objects
        else:
            eval_context["messages"] = messages

        # Include the state that was passed to the reward function
        eval_context["state"] = reward_function_inputs.get("state")
        eval_context["task_achieved"] = reward_function_inputs.get(
            "task_achieved", False
        )
        eval_context["task_definition_name"] = reward_function_inputs.get(
            "task_definition_name", ""
        )

        # Include ground truth if available
        if reward_function_inputs.get("ground_truth"):
            eval_context["ground_truth"] = reward_function_inputs["ground_truth"]

        return eval_context

    else:
        # Fallback: reconstruct from step outputs (for older trajectory files)
        eval_context: Dict[str, Any] = {}

        # Create dummy messages for older format
        dummy_messages = [
            Message(role="assistant", content="Agent completed the game"),
            Message(role="system", content="You reached the goal! Congratulations!"),
        ]
        eval_context["messages"] = dummy_messages

        # Reconstruct state from step outputs
        if "step_outputs" in trajectory:
            successful_func_calls = [[]]  # Single turn for simplicity
            for step_data in trajectory["step_outputs"]:
                # Extract action from the reason field
                reason = step_data.get("reason", "")
                action = "unknown"
                if "Agent took action:" in reason:
                    action = reason.split("Agent took action:")[-1].strip()

                func_call = {"args": {"action": action}}
                successful_func_calls[0].append(func_call)

            eval_context["state"] = {"successful_func_calls": successful_func_calls}

        # Add fallback fields
        eval_context.update(
            {
                "task_achieved": False,
                "task_definition_name": trajectory.get("task_id", ""),
            }
        )

        return eval_context
