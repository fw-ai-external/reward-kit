import argparse
import json
import logging

try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS_LIB = True
except ImportError:
    HAS_DATASETS_LIB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import math_reward functions from reward_kit
try:
    from reward_kit.rewards.math import math_reward as numeric_math_reward
    from reward_kit.rewards.multiple_choice_math_reward import multiple_choice_math_reward
    from reward_kit.rewards.list_comparison_math_reward import list_comparison_math_reward
    # from reward_kit.models import Message # Rewards can handle dicts
    HAS_REWARD_KIT_MATH_FUNCTIONS = True
except ImportError as e:
    logger.error(f"Failed to import one or more math reward functions from reward_kit.rewards: {e}. Make sure reward-kit is installed and accessible.")
    HAS_REWARD_KIT_MATH_FUNCTIONS = False


def convert_math_dataset_to_openai_jsonl(
    dataset_name: str,
    output_file_path: str,
    query_column: str = "query",
    solution_column_for_assistant: str = "solution", # Column for assistant message
    ground_truth_answer_column: str = "answer",   # Column for the verifiable answer
    split: str = "train",
    filter_by_match: bool = False, # New flag to control filtering
    math_type: str = "numeric" # 'numeric', 'mcq', 'list'
):
    """
    Loads a HuggingFace math dataset.
    Optionally filters rows by comparing an answer extracted from 'solution_column_for_assistant'
    against 'ground_truth_answer_column' using a specified reward_kit math_reward function.
    Converts kept rows into an OpenAI-style messages JSONL format.

    Output format if not filtering, or if filtering and match is found:
    {
      "messages": [
        {"role": "user", "content": "<content_from_query_column>"},
        {"role": "assistant", "content": "<content_from_solution_column_for_assistant>"}
      ],
      "ground_truth_answer_from_column": "<content_from_ground_truth_answer_column>",
      "match_details": { // Only if filter_by_match is True
          "filter_passed": true/false,
          "reward_score": score_from_math_reward
      }
    }

    Args:
        dataset_name: Name or path of the HuggingFace dataset.
        output_file_path: Path to save the output JSONL file.
        query_column: Column for the user message content.
        solution_column_for_assistant: Column for the assistant message content (e.g., detailed solution).
        ground_truth_answer_column: Column for the concise ground truth answer (for comparison).
        split: Dataset split to process.
        filter_by_match: If True, use the specified math_reward to compare
                         solution_column_for_assistant with ground_truth_answer_column
                         and only keep matching rows.
        math_type: Type of math reward to use for filtering ('numeric', 'mcq', 'list').
    """
    if not HAS_DATASETS_LIB:
        logger.error("The 'datasets' library is not installed. Please install it with 'pip install datasets'.")
        return
    if filter_by_match and not HAS_REWARD_KIT_MATH_FUNCTIONS:
        logger.error("Filtering by match requires math reward functions from reward_kit, which could not be imported.")
        return

    logger.info(f"Loading dataset '{dataset_name}', split '{split}'...")
    try:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        return

    logger.info(f"Dataset loaded. Features: {dataset.features}")

    required_cols = {query_column, solution_column_for_assistant, ground_truth_answer_column}
    for col_name in required_cols:
        if col_name not in dataset.column_names:
            logger.error(f"Required column '{col_name}' not found in dataset. Available columns: {dataset.column_names}")
            return

    logger.info(f"Processing {len(dataset)} examples. Output to: {output_file_path}. Filtering by match: {filter_by_match}, Math Type: {math_type}")
    
    kept_count = 0
    processed_count = 0
    
    reward_function_to_use = None
    if filter_by_match:
        if math_type == "numeric":
            reward_function_to_use = numeric_math_reward
            logger.info("Using numeric_math_reward for filtering.")
        elif math_type == "mcq":
            reward_function_to_use = multiple_choice_math_reward
            logger.info("Using multiple_choice_math_reward for filtering.")
        elif math_type == "list":
            reward_function_to_use = list_comparison_math_reward
            logger.info("Using list_comparison_math_reward for filtering.")
        else:
            logger.error(f"Invalid math_type '{math_type}'. Cannot perform filtering. Exiting.")
            return
        
        if reward_function_to_use is None: # Should be caught by HAS_REWARD_KIT_MATH_FUNCTIONS earlier if import failed
             logger.error(f"Reward function for math_type '{math_type}' could not be resolved. Exiting.")
             return

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for example in dataset:
            processed_count += 1
            try:
                query_content = str(example.get(query_column, ""))
                solution_content = str(example.get(solution_column_for_assistant, "")) # For assistant msg & extraction
                gt_answer_content = str(example.get(ground_truth_answer_column, "")) # For comparison & as extra field

                if not query_content:
                    logger.warning(f"Skipping example due to missing query content: {example.get('uuid', 'N/A')}")
                    continue
                
                match_passed = True # Assume pass if not filtering
                reward_score = None
                eval_result = None 

                if filter_by_match and reward_function_to_use:
                    if not solution_content or not gt_answer_content:
                        logger.warning(f"Skipping filtering for example due to missing solution or GT answer: {example.get('uuid', 'N/A')}")
                        match_passed = False # Cannot compare, so filter out
                    else:
                        dummy_generated_messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": solution_content}]
                        dummy_original_messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": gt_answer_content}]
                        
                        # Call the selected reward function
                        eval_result_obj = reward_function_to_use(
                            messages=dummy_generated_messages,
                            original_messages=dummy_original_messages
                        )
                        # eval_result is now an EvaluateResult object, convert to dict for existing logic
                        eval_result = eval_result_obj

                        reward_score = eval_result['score']
                        match_passed = reward_score >= 0.999 # Using a high threshold for "match"
                        
                        if not match_passed:
                            logger.debug(f"Row discarded ({math_type}): Solution-Answer mismatch. Score: {reward_score:.3f}. GT: '{gt_answer_content[:50]}...', Sol: '{solution_content[:50]}...'")

                if not filter_by_match or match_passed:
                    messages = [
                        {"role": "user", "content": query_content},
                        {"role": "assistant", "content": solution_content} 
                    ]
                    
                    output_data = {
                        "messages": messages,
                        "ground_truth_answer_from_column": gt_answer_content
                    }

                    if filter_by_match and eval_result:
                        metrics_data = eval_result.get('metrics', {})
                        match_reason = eval_result.get('reason', 'N/A')
                        
                        # Adjust metric keys based on math_type for more specific details
                        sol_extraction_reason = "N/A"
                        gt_extraction_reason = "N/A"

                        if math_type == "numeric":
                            sol_extraction_metric = metrics_data.get("extracted_generated_answers")
                            gt_extraction_metric = metrics_data.get("extracted_original_answers")
                        elif math_type == "mcq":
                            sol_extraction_metric = metrics_data.get("extracted_generated_mcq")
                            gt_extraction_metric = metrics_data.get("extracted_original_mcq")
                        elif math_type == "list":
                            sol_extraction_metric = metrics_data.get("extracted_generated_lists")
                            gt_extraction_metric = metrics_data.get("extracted_original_lists")
                        else: # Fallback, should not happen if validated before
                            sol_extraction_metric = None
                            gt_extraction_metric = None
                        
                        if sol_extraction_metric and isinstance(sol_extraction_metric, dict):
                            sol_extraction_reason = sol_extraction_metric.get('reason', "N/A")
                        if gt_extraction_metric and isinstance(gt_extraction_metric, dict):
                            gt_extraction_reason = gt_extraction_metric.get('reason', "N/A")

                        output_data["match_details"] = {
                            "filter_passed": True,
                            "reward_score": reward_score,
                            "match_comparison_reason": match_reason,
                            "math_type_used_for_filter": math_type,
                            "extracted_from_solution_column": sol_extraction_reason,
                            "extracted_from_gt_answer_column": gt_extraction_reason
                        }
                    
                    outfile.write(json.dumps(output_data) + '\n')
                    kept_count += 1

                if processed_count % 10000 == 0:
                    logger.info(f"Processed {processed_count} examples, Kept: {kept_count}...")

            except Exception as e:
                logger.error(f"Error processing example (UUID: {example.get('uuid', 'N/A')}): {e}", exc_info=True)
    
    logger.info(f"Successfully processed {processed_count} examples. Kept and converted {kept_count} examples from '{dataset_name}' to '{output_file_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace math dataset to OpenAI-style messages JSONL, with optional filtering."
    )
    parser.add_argument("dataset_name", type=str, help="HF dataset name (e.g., 'open-r1/OpenR1-Math-220k').")
    parser.add_argument("output_file_path", type=str, help="Path for the output JSONL file.")
    
    parser.add_argument("--query_column", type=str, default="problem", help="Dataset column for user query (default: 'problem').")
    parser.add_argument("--solution_column_for_assistant", type=str, default="solution", help="Dataset column for assistant's detailed solution (default: 'solution').")
    parser.add_argument("--ground_truth_answer_column", type=str, default="answer", help="Dataset column for concise ground truth answer (default: 'answer').")
    
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: 'train').")
    parser.add_argument("--filter_by_match", action="store_true", help="Enable filtering: only keep rows where answer extracted from solution matches ground_truth_answer.")
    parser.add_argument("--math_type", type=str, default="numeric", choices=["numeric", "mcq", "list"], help="Type of math reward to use for filtering (default: 'numeric').")
    
    args = parser.parse_args()

    if not HAS_REWARD_KIT_MATH_FUNCTIONS and args.filter_by_match:
        logger.error("Cannot perform filtering because 'math reward functions' from reward_kit could not be imported. Exiting.")
    else:
        convert_math_dataset_to_openai_jsonl(
            dataset_name=args.dataset_name,
            output_file_path=args.output_file_path,
            query_column=args.query_column,
            solution_column_for_assistant=args.solution_column_for_assistant,
            ground_truth_answer_column=args.ground_truth_answer_column,
            split=args.split,
            filter_by_match=args.filter_by_match,
            math_type=args.math_type
        )

    # Example usage from CLI:
    # Filtered examples:
    # python scripts/convert_hf_math_to_openai_jsonl.py open-r1/OpenR1-Math-220k openr1_numeric.jsonl --filter_by_match --math_type numeric
    # python scripts/convert_hf_math_to_openai_jsonl.py open-r1/OpenR1-Math-220k openr1_mcq.jsonl --filter_by_match --math_type mcq
    # python scripts/convert_hf_math_to_openai_jsonl.py open-r1/OpenR1-Math-220k openr1_list.jsonl --filter_by_match --math_type list
    # Unfiltered (retains old structure for comparison, but with new field names):
    # python scripts/convert_hf_math_to_openai_jsonl.py open-r1/OpenR1-Math-220k openr1_unfiltered.jsonl
