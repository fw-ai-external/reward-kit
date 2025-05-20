import argparse
import json
import logging
import math  # Added import
import re  # Added import

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


def convert_math_dataset_to_openai_jsonl(
    dataset_name: str,
    output_file_path: str,
    query_column: str = "query",
    solution_column_for_assistant: str = "solution",  # Column for assistant message
    ground_truth_answer_column: str = "answer",  # Column for the verifiable answer
    split: str = "train",
    filter_by_match: bool = False,
    math_type: str = "numeric",
    config_name: str = None,
):
    """
    Loads a HuggingFace math dataset.
    Optionally filters rows by comparing an answer extracted from 'solution_column_for_assistant'
    against 'ground_truth_answer_column' using a specified reward_kit math_reward function.
    If math_type is 'numeric', an additional check ensures 'ground_truth_answer_column' is strictly numeric.
    Converts kept rows into an OpenAI-style messages JSONL format.

    Output format:
    {
      "messages": [
        {"role": "user", "content": "<content_from_query_column>"},
        {"role": "assistant", "content": "<content_from_solution_column_for_assistant>"}
      ],
      "ground_truth_answer_from_column": "<content_from_ground_truth_answer_column>",
      "match_details": { // Only if filter_by_match is True
          "filter_passed": true/false,
          "reward_score": score_from_math_reward_or_strict_check,
          "match_comparison_reason": "Reason for pass/fail",
          "math_type_used_for_filter": "numeric/mcq/list",
          "extracted_from_solution_column": "Extraction details from solution",
          "extracted_from_gt_answer_column": "Extraction details from ground truth"
      }
    }
    """
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
        dataset = load_dataset(
            dataset_name, name=config_name, split=split, trust_remote_code=True
        )
    except Exception as e:
        logger.error(
            f"Failed to load dataset '{dataset_name}' (config: {config_name}): {e}"
        )
        return

    logger.info(f"Dataset loaded. Features: {dataset.features}")

    required_cols = {
        query_column,
        solution_column_for_assistant,
        ground_truth_answer_column,
    }
    for col_name in required_cols:
        if col_name not in dataset.column_names:
            logger.error(
                f"Required column '{col_name}' not found in dataset. Available columns: {dataset.column_names}"
            )
            return

    logger.info(
        f"Processing {len(dataset)} examples. Output to: {output_file_path}. Filtering by match: {filter_by_match}, Math Type: {math_type}"
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

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for example in dataset:
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
                                and math.isclose(sol_val, gt_num_val)
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
                                    if math.isclose(num_after_hash_val, gt_num_val):
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
                    "ground_truth_answer_from_column": final_ground_truth_for_jsonl,
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
        f"Successfully processed {processed_count} examples. Kept and converted {kept_count} examples from '{dataset_name}' to '{output_file_path}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace math dataset to OpenAI-style messages JSONL, with optional filtering."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HF dataset name (e.g., 'open-r1/OpenR1-Math-220k').",
    )
    parser.add_argument(
        "output_file_path", type=str, help="Path for the output JSONL file."
    )

    parser.add_argument(
        "--query_column",
        type=str,
        default="problem",
        help="Dataset column for user query (default: 'problem').",
    )
    parser.add_argument(
        "--solution_column_for_assistant",
        type=str,
        default="solution",
        help="Dataset column for assistant's detailed solution (default: 'solution').",
    )
    parser.add_argument(
        "--ground_truth_answer_column",
        type=str,
        default="answer",
        help="Dataset column for concise ground truth answer (default: 'answer').",
    )

    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: 'train')."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Dataset config name (e.g., 'main' for gsm8k).",
    )
    parser.add_argument(
        "--filter_by_match",
        action="store_true",
        help="Enable filtering: only keep rows where answer extracted from solution matches ground_truth_answer. For 'numeric' math_type, ground_truth_answer must also be strictly numeric.",
    )
    parser.add_argument(
        "--math_type",
        type=str,
        default="numeric",
        choices=["numeric", "mcq", "list"],
        help="Type of math reward to use for filtering (default: 'numeric').",
    )

    args = parser.parse_args()

    if not HAS_REWARD_KIT_MATH_FUNCTIONS and args.filter_by_match:
        logger.error(
            "Cannot perform filtering because 'math reward functions' or 'MetricResult' from reward_kit could not be imported. Exiting."
        )
    else:
        convert_math_dataset_to_openai_jsonl(
            dataset_name=args.dataset_name,
            output_file_path=args.output_file_path,
            query_column=args.query_column,
            solution_column_for_assistant=args.solution_column_for_assistant,
            ground_truth_answer_column=args.ground_truth_answer_column,
            split=args.split,
            filter_by_match=args.filter_by_match,
            math_type=args.math_type,
            config_name=args.config_name,
        )
