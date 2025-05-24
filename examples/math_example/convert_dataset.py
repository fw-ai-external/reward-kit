# mypy: ignore-errors
# pylint: disable=all
import json
import logging
import math  # Added import
import os  # Added for Hydra path management
import re  # Added import
import sys  # Added to fix flake8 error
from typing import Optional  # Added Optional

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    from datasets import Dataset, load_dataset

    HAS_DATASETS_LIB = True
except ImportError:
    HAS_DATASETS_LIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Attempt to import math_reward functions from reward_kit
try:
    from reward_kit.models import MetricResult  # Import MetricResult for type checking
    from reward_kit.rewards.list_comparison_math_reward import (
        list_comparison_math_reward,
    )
    from reward_kit.rewards.math import extract_numbers  # Added direct import
    from reward_kit.rewards.math import math_reward as numeric_math_reward
    from reward_kit.rewards.multiple_choice_math_reward import (
        multiple_choice_math_reward,
    )

    HAS_REWARD_KIT_MATH_FUNCTIONS = True
except ImportError as e:
    logger.error(
        f"Failed to import one or more math reward functions or MetricResult from reward_kit: {e}. Make sure reward-kit is installed and accessible."
    )
    HAS_REWARD_KIT_MATH_FUNCTIONS = False


# Regex for is_strictly_numeric function
_NUM_REGEX_STR = r"-?\d+(\.\d+)?"
_STRICTLY_NUMERIC_REGEX_STR = rf"^\s*({_NUM_REGEX_STR}(\s*,\s*{_NUM_REGEX_STR})*)?\s*$"
_STRICTLY_NUMERIC_COMPILED_REGEX = re.compile(_STRICTLY_NUMERIC_REGEX_STR)

# Regex for MCQ detection
# This regex looks for patterns like A), (A), A., [A] that are preceded by a space or start of line,
# and are followed by some text that looks like an option.
# More aggressive MCQ_PATTERN_REGEX to catch more formats including TeX and numerical options.
MCQ_PATTERN_REGEX = re.compile(
    r"(\b(?:[A-Za-z]|[0-9]+)\s*[\)\.])|"  # Matches A) or 1.
    r"([\(\[]\s*(?:[A-Za-z]|[0-9]+)\s*[\)\]])|"  # Matches (A) or [1]
    r"(\(\s*\\mathrm\s*\{\s*[A-Za-z]\s*\}\s*\))"  # Matches (\mathrm{A})
)

# Regex for single letter choice detection (e.g., "A", "(B)", "C.")
SINGLE_LETTER_CHOICE_REGEX = re.compile(r"^\s*\(?\s*([A-Z])\s*\)?\s*\.?\s*$")


def is_multiple_choice_question(text: str) -> bool:
    """
    Detects if the text likely contains a multiple-choice question format.
    Searches for patterns like A), (B), C. at the start of lines.
    """
    if MCQ_PATTERN_REGEX.search(text):
        return True
    return False


def is_strictly_numeric(text: str) -> bool:
    """
    Checks if a string represents one or more simple numbers (comma-separated list is allowed).
    Allows integers and floats. Disallows other characters like letters, LaTeX, etc.
    An empty string or a string with only whitespace is not considered strictly numeric.
    """
    if not text or text.isspace():
        return False
    return bool(_STRICTLY_NUMERIC_COMPILED_REGEX.fullmatch(text))


def convert_math_dataset_to_openai_jsonl(cfg: DictConfig):
    """
    Loads a HuggingFace math dataset based on Hydra configuration.
    This script is intended to be run using Hydra. Example:
    `python examples/math_example/convert_dataset.py dataset.name=gsm8k output.file_path=gsm8k_converted.jsonl`
    Optionally filters rows by comparing an answer extracted from 'cfg.dataset.solution_column_for_assistant'
    against 'cfg.dataset.ground_truth_answer_column' using a specified reward_kit math_reward function.
    If cfg.processing.math_type is 'numeric', an additional check ensures 'cfg.dataset.ground_truth_answer_column' is strictly numeric.
    Converts kept rows into an OpenAI-style messages JSONL format.

    Output format is written to cfg.output.file_path.
    """
    # Extract parameters from Hydra config
    dataset_name = cfg.dataset.name
    output_file_path = (
        cfg.output.file_path
    )  # This will be relative to Hydra's run dir if not absolute
    query_column = cfg.dataset.query_column
    solution_column_for_assistant = cfg.dataset.solution_column_for_assistant
    ground_truth_answer_column = cfg.dataset.ground_truth_answer_column
    split = cfg.dataset.split
    filter_by_match = cfg.processing.filter_by_match
    math_type = cfg.processing.math_type
    config_name = cfg.dataset.config_name

    # If output_file_path is relative, it's relative to the Hydra output directory.
    # If it needs to be relative to the original CWD or an absolute path is preferred,
    # one might use: output_file_path = hydra.utils.to_absolute_path(cfg.output.file_path)
    # For simplicity, we'll assume cfg.output.file_path is handled as needed (e.g., absolute or relative to hydra's dir).
    # Hydra changes CWD to its output dir, so relative paths in cfg.output.file_path will be created there.

    if not HAS_DATASETS_LIB:
        logger.error(
            "The 'datasets' library is not installed. Please install it with 'pip install datasets'."
        )
        return
    if filter_by_match and not HAS_REWARD_KIT_MATH_FUNCTIONS:
        logger.error(
            "Filtering by match requires math reward functions and MetricResult from reward_kit, which could not be imported."
        )
        return

    logger.info(
        f"Loading dataset '{dataset_name}' (config: {config_name}), split '{split}'..."
    )
    try:
        dataset_hf = load_dataset(  # Renamed to avoid conflict with cfg.dataset
            dataset_name, name=config_name, split=split, trust_remote_code=True
        )
    except Exception as e:
        logger.error(
            f"Failed to load dataset '{dataset_name}' (config: {config_name}): {e}"
        )
        return

    logger.info(f"Dataset loaded. Features: {dataset_hf.features}")

    required_cols = {
        query_column,
        solution_column_for_assistant,
        ground_truth_answer_column,
    }
    for col_name in required_cols:
        if col_name not in dataset_hf.column_names:
            logger.error(
                f"Required column '{col_name}' not found in dataset. Available columns: {dataset_hf.column_names}"
            )
            return

    logger.info(
        f"Processing {len(dataset_hf)} examples. Output to: {output_file_path}. "
        f"Filtering by match: {filter_by_match}, Math Type: {math_type}"
    )

    kept_count = 0
    processed_count = 0

    reward_function_to_use = None
    if filter_by_match:
        if math_type == "numeric":
            reward_function_to_use = numeric_math_reward
            logger.info(
                "Using numeric_math_reward for filtering with strict ground truth check."
            )
        elif math_type == "mcq":
            reward_function_to_use = multiple_choice_math_reward
            logger.info("Using multiple_choice_math_reward for filtering.")
        elif math_type == "list":
            reward_function_to_use = list_comparison_math_reward
            logger.info("Using list_comparison_math_reward for filtering.")
        else:
            logger.error(
                f"Invalid math_type '{math_type}'. Cannot perform filtering. Exiting."
            )
            return

        if reward_function_to_use is None:
            logger.error(
                f"Reward function for math_type '{math_type}' could not be resolved. Exiting."
            )
            return

    # Ensure output directory exists if path is relative (Hydra usually handles this for its own dir)
    # If output_file_path is absolute, the directory must exist or be creatable.
    if not os.path.isabs(output_file_path):
        # output_file_path is relative to hydra.run.dir, which is the CWD.
        # os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # Not strictly needed if file is in CWD
        pass  # Hydra manages its output directory.

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for example in dataset_hf:  # Iterate over dataset_hf
            processed_count += 1
            try:
                query_content = str(example.get(query_column, ""))
                solution_content = str(example.get(solution_column_for_assistant, ""))
                gt_answer_content = str(example.get(ground_truth_answer_column, ""))

                if not query_content:
                    logger.warning(
                        f"Skipping example due to missing query content: {example.get('uuid', 'N/A')}"
                    )
                    continue

                # New MCQ Filtering Step
                if is_multiple_choice_question(query_content):
                    logger.info(
                        f"Skipping example {example.get('uuid', 'N/A')} as query appears to be a multiple-choice question."
                    )
                    continue

                # New filter: If math_type is numeric, but ground_truth_answer_column suggests a lettered choice
                if math_type == "numeric":
                    if SINGLE_LETTER_CHOICE_REGEX.fullmatch(gt_answer_content):
                        logger.info(
                            f"Skipping example {example.get('uuid', 'N/A')} for 'numeric' math_type as ground_truth_answer_column ('{gt_answer_content}') suggests a lettered choice."
                        )
                        continue

                messages = [
                    {"role": "user", "content": query_content},
                    {
                        "role": "assistant",
                        "content": "PLACEHOLDER_ASSISTANT_CONTENT",
                    },  # Will be replaced
                ]

                # Initialize final content variables
                final_assistant_content = solution_content
                final_ground_truth_for_jsonl = gt_answer_content

                if (
                    HAS_REWARD_KIT_MATH_FUNCTIONS
                ):  # extract_numbers is from reward_kit.rewards.math
                    orig_gt_extracts = extract_numbers(  # Use direct import
                        gt_answer_content
                    )

                    if len(orig_gt_extracts) == 1 and isinstance(
                        orig_gt_extracts[0][1], (float, int)
                    ):
                        _, gt_num_val = orig_gt_extracts[0]
                        gt_num_str_for_box = (
                            str(int(gt_num_val))
                            if gt_num_val == int(gt_num_val)
                            else str(gt_num_val)
                        )
                        boxed_gt_string = f"\\boxed{{{gt_num_str_for_box}}}"

                        final_ground_truth_for_jsonl = (
                            boxed_gt_string  # Update ground truth format
                        )

                        # Check if solution_content already has the correct boxed answer
                        sol_extracts = extract_numbers(  # Use direct import
                            solution_content
                        )
                        already_correctly_boxed = False
                        for sol_text, sol_val in sol_extracts:
                            if (
                                sol_text == boxed_gt_string
                                and isinstance(sol_val, (float, int))
                                # Ensure gt_num_val is float for isclose
                                and isinstance(gt_num_val, (float, int))
                                and math.isclose(sol_val, float(gt_num_val))
                            ):
                                already_correctly_boxed = True
                                break

                        if not already_correctly_boxed:
                            temp_solution_content = solution_content
                            replaced_via_hash = False
                            # Regex to find all "#### <number>" patterns
                            hash_patterns = list(
                                re.finditer(
                                    r"(####\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?))",
                                    temp_solution_content,
                                )
                            )

                            for match in sorted(
                                hash_patterns, key=lambda m: m.end(1), reverse=True
                            ):
                                num_str_after_hash = match.group(2)
                                try:
                                    num_after_hash_val = float(
                                        num_str_after_hash.replace(",", "")
                                    )
                                    # Ensure gt_num_val is float for isclose
                                    assert isinstance(gt_num_val, (float, int))
                                    if math.isclose(
                                        num_after_hash_val, float(gt_num_val)
                                    ):
                                        start, end = match.span(
                                            1
                                        )  # Get span of "#### <number>"
                                        temp_solution_content = (
                                            temp_solution_content[:start]
                                            + boxed_gt_string
                                            + temp_solution_content[end:]
                                        )
                                        replaced_via_hash = True
                                        break
                                except ValueError:
                                    continue

                            if replaced_via_hash:
                                final_assistant_content = temp_solution_content
                            else:
                                final_assistant_content = (
                                    solution_content
                                    + f"\n\nThe final answer is {boxed_gt_string}."
                                )
                        # If already_correctly_boxed, final_assistant_content remains original solution_content
                        # which should contain the correct boxed string.

                messages[1][
                    "content"
                ] = final_assistant_content  # Update assistant message

                output_data = {
                    "messages": messages,
                    "ground_truth": final_ground_truth_for_jsonl,
                }

                should_keep_row = True  # Default if not filtering

                if filter_by_match:
                    details_filter_passed = False
                    details_reward_score = 0.0
                    details_match_reason = "Filter prerequisites not met (e.g., missing content or reward function error)."
                    details_sol_extraction = "N/A"
                    details_gt_extraction = "N/A"

                    if not final_assistant_content or not final_ground_truth_for_jsonl:
                        logger.warning(
                            f"Cannot filter example {example.get('uuid', 'N/A')} due to missing final assistant content or final ground truth."
                        )
                        details_match_reason = "Missing final assistant content or final ground truth for comparison."
                    elif reward_function_to_use:
                        try:
                            # Use final_assistant_content and final_ground_truth_for_jsonl for filtering
                            dummy_generated_messages = [
                                {"role": "user", "content": "Q"},
                                {
                                    "role": "assistant",
                                    "content": final_assistant_content,
                                },
                            ]
                            # dummy_original_messages is not strictly needed if ground_truth param is used directly

                            eval_result_obj = reward_function_to_use(
                                messages=dummy_generated_messages,
                                ground_truth=final_ground_truth_for_jsonl,  # Use modified ground truth
                            )

                            details_reward_score = eval_result_obj.score
                            details_filter_passed = details_reward_score >= 0.999
                            details_match_reason = (
                                eval_result_obj.reason
                                if eval_result_obj.reason is not None
                                else "N/A"
                            )

                            metrics_data = (
                                eval_result_obj.metrics
                                if eval_result_obj.metrics is not None
                                else {}
                            )

                            metric_key_gen = ""
                            metric_key_orig = ""
                            if math_type == "numeric":
                                metric_key_gen = "extracted_generated_answers"
                                metric_key_orig = "extracted_original_answers"
                            elif math_type == "mcq":
                                metric_key_gen = "extracted_generated_mcq"
                                metric_key_orig = "extracted_original_mcq"
                            elif math_type == "list":
                                metric_key_gen = "extracted_generated_lists"
                                metric_key_orig = "extracted_original_lists"

                            if metric_key_gen:
                                sol_extraction_metric = metrics_data.get(metric_key_gen)
                                if sol_extraction_metric and isinstance(
                                    sol_extraction_metric, MetricResult
                                ):
                                    details_sol_extraction = (
                                        sol_extraction_metric.reason
                                    )

                            if metric_key_orig:
                                gt_extraction_metric = metrics_data.get(metric_key_orig)
                                if gt_extraction_metric and isinstance(
                                    gt_extraction_metric, MetricResult
                                ):
                                    details_gt_extraction = gt_extraction_metric.reason

                            if not details_filter_passed:
                                logger.debug(
                                    f"Row for example {example.get('uuid', 'N/A')} discarded ({math_type}): Solution-Answer mismatch. Score: {details_reward_score:.3f}. Reason: {details_match_reason}. GT: '{gt_answer_content[:100]}...', Sol: '{solution_content[:100]}...'"
                                )

                        except Exception as e_reward:
                            logger.error(
                                f"Error during reward function execution for example {example.get('uuid', 'N/A')}: {e_reward}",
                                exc_info=True,
                            )
                            details_match_reason = (
                                f"Error in reward function: {e_reward}"
                            )
                            details_filter_passed = (
                                False  # Ensure it's marked as failed
                            )
                    else:  # Should not be reached if initial checks are correct
                        details_match_reason = (
                            "Reward function not configured for filtering."
                        )

                    output_data["match_details"] = {
                        "filter_passed": details_filter_passed,
                        "reward_score": details_reward_score,
                        "match_comparison_reason": details_match_reason,
                        "math_type_used_for_filter": math_type,
                        "extracted_from_solution_column": details_sol_extraction,
                        "extracted_from_gt_answer_column": details_gt_extraction,
                    }
                    should_keep_row = details_filter_passed

                if should_keep_row:
                    outfile.write(json.dumps(output_data) + "\n")
                    kept_count += 1

                if processed_count % 10000 == 0:
                    logger.info(
                        f"Processed {processed_count} examples, Kept: {kept_count}..."
                    )

            except Exception as e:
                logger.error(
                    f"Error processing example (UUID: {example.get('uuid', 'N/A')}): {e}",
                    exc_info=True,
                )

    logger.info(
        f"Successfully processed {processed_count} examples. Kept and converted {kept_count} examples from '{cfg.dataset.name}' to '{output_file_path}'."
    )
    logger.info(f"Output written to: {os.path.abspath(output_file_path)}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_conversion(cfg: DictConfig) -> None:
    """
    Main entry point for the script, configured by Hydra.
    """
    logger.info("Starting dataset conversion process with Hydra configuration...")
    logger.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Call the main conversion function with the Hydra config
    # The convert_math_dataset_to_openai_jsonl function now directly accepts the cfg object.
    convert_math_dataset_to_openai_jsonl(cfg)

    logger.info("Dataset conversion process finished.")


if __name__ == "__main__":
    # Ensure Hydra is installed
    try:
        import hydra
    except ImportError:
        logger.error(
            "Hydra is not installed. Please install it with 'pip install hydra-core'."
        )
        sys.exit(1)

    run_conversion()
