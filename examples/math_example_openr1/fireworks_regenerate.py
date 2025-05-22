# mypy: ignore-errors
# pylint: disable=all
import argparse
import asyncio
import json
import os
import sys

import aiohttp

# import logging  # Added for new utility (now removed as unused)


# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any, Dict, List, Optional

from reward_kit.common_utils import load_jsonl  # Import the new utility
from reward_kit.models import Message
from reward_kit.rewards.math import math_reward

# Configure basic logging if you want to see logs from load_jsonl
# logging.basicConfig(level=logging.INFO)

# Configuration for Fireworks API
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL_NAME = (
    "accounts/fireworks/models/qwen3-30b-a3b"  # Using the same model as gsm8k example
)
RECORDED_DATA_FILENAME_OPENR1 = (
    "fireworks_regenerate_recorded_data_openr1.jsonl"  # Specific filename
)


# Removed local load_dataset function


async def generate_with_fireworks_inner(
    session: aiohttp.ClientSession,
    user_prompt: str,
    api_key: str,
    force_live_api: bool = False,
    recorded_data_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Core logic to generate a response using Fireworks API or mock data, asynchronously."""

    if not force_live_api and os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true":
        if recorded_data_map and user_prompt in recorded_data_map:
            print(
                f"Mocking Fireworks API call for regeneration using recorded data for prompt: '{user_prompt[:70]}...'"
            )
            return recorded_data_map[user_prompt]
        else:
            print(
                f"Warning: Mock data not found for prompt: '{user_prompt[:70]}...'. Using fallback mock response or None."
            )
            return "A generic mocked math solution for OpenR1 that will likely result in a score of 0."

    # there are some examples with multiple choice and model just answers with multiple choice instead.
    system_prompt = "IMPORTANT: You MUST provide your final numerical answer enclosed *only* in `\\boxed{answer}`. Do not include any other numbers or text within the box. Your entire response should be your reasoning, followed by the single, final boxed answer. Example: `\\boxed{123.45}`. If the question is a multiple choice question, please still provide the numerical answer."
    payload: Dict[str, Any] = {
        "model": FIREWORKS_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 8000,
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(
            FIREWORKS_API_URL,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as response:
            response.raise_for_status()
            completion = await response.json()
            if completion.get("choices") and len(completion["choices"]) > 0:
                return completion["choices"][0].get("message", {}).get("content")
            else:
                print(
                    f"Warning: No choices returned from Fireworks API. Response: {completion}"
                )
                return None
    except aiohttp.ClientError as e:
        print(f"Error calling Fireworks API (aiohttp): {e}")
        if hasattr(e, "status"):
            print(f"Status: {e.status}, Message: {e.message}")  # type: ignore
        return None
    except asyncio.TimeoutError:
        print(
            f"Timeout error calling Fireworks API for prompt: '{user_prompt[:70]}...'"
        )
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding Fireworks API response: {e}")
        return None


async def generate_with_fireworks(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    user_prompt: str,
    api_key: str,
    force_live_api: bool = False,
    recorded_data_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Asynchronous wrapper for generate_with_fireworks_inner with semaphore control."""
    async with semaphore:
        if not (
            not force_live_api and os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true"
        ):
            try:
                # Attempt to get semaphore details safely
                # This part is for logging and might not be perfectly accurate for all executor types
                concurrency_str = f"{10 - semaphore._value}/10 (approx)"
            except AttributeError:
                concurrency_str = "N/A"
            print(
                f"Calling Fireworks API for prompt: '{user_prompt[:70]}...' (Concurrency: {concurrency_str})"
            )

        return await generate_with_fireworks_inner(
            session, user_prompt, api_key, force_live_api, recorded_data_map
        )


async def main(args):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    recorded_data_path = os.path.join(
        os.path.dirname(__file__), RECORDED_DATA_FILENAME_OPENR1
    )  # Use specific filename

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    api_key = os.environ.get("FIREWORKS_API_KEY")
    is_mock_mode_via_env = os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true"
    recorded_data_map_for_mocking: Optional[Dict[str, str]] = None

    if not args.regenerate_recorded_data and not is_mock_mode_via_env and not api_key:
        print(
            "Error: FIREWORKS_API_KEY environment variable is not set (and not in mock/regenerate mode)."
        )
        return

    if args.regenerate_recorded_data and not api_key:
        print(
            "Error: FIREWORKS_API_KEY must be set when using --regenerate-recorded-data."
        )
        return

    dataset = load_jsonl(dataset_path)
    all_passed = True
    passed_samples = 0

    live_call_results_to_record: List[Dict] = []

    if args.regenerate_recorded_data:
        print(
            f"MODE: Regenerating recorded data for OpenR1. Live API calls will be made. Output to: {recorded_data_path}"
        )
    elif is_mock_mode_via_env:
        print(
            "MODE: Attempting to use mock data for Fireworks API calls from recorded file for OpenR1."
        )
        if os.path.exists(recorded_data_path):
            try:
                recorded_data_list = load_jsonl(recorded_data_path)
                recorded_data_map_for_mocking = {
                    record["user_prompt"]: record["regenerated_content"]
                    for record in recorded_data_list
                    if "user_prompt" in record and "regenerated_content" in record
                }
                if recorded_data_map_for_mocking:
                    print(
                        f"Successfully loaded {len(recorded_data_map_for_mocking)} records from {recorded_data_path} for mocking."
                    )
                else:
                    print(
                        f"Warning: No valid records found in {recorded_data_path} to use for mocking."
                    )
            except Exception as e:
                print(
                    f"Warning: Could not load or parse {recorded_data_path} for mocking: {e}"
                )
        else:
            print(
                f"Warning: Mock mode enabled, but recorded data file {recorded_data_path} not found."
            )
    else:
        print(
            "MODE: Using live Fireworks API calls for OpenR1 (not saving to recorded data file)."
        )

    print(
        f"Starting Fireworks Regeneration & Evaluation for OpenR1 Math Example using {dataset_path}...\n"  # Modified
    )

    initial_samples_with_indices = [
        {"index": i, "data": item} for i, item in enumerate(dataset)
    ]

    if args.indices:
        try:
            selected_indices = {int(idx.strip()) for idx in args.indices.split(",")}
            samples_to_process_with_indices = [
                s_info
                for s_info in initial_samples_with_indices
                if s_info["index"] in selected_indices
            ]
            if not samples_to_process_with_indices:
                print(
                    f"Warning: No samples found for specified indices: {args.indices}. Exiting."
                )
                return
            print(
                f"Processing specific indices: {sorted(list(s['index'] for s in samples_to_process_with_indices))}"
            )
        except ValueError:
            print(
                f"Error: Invalid format for --indices. Please provide comma-separated integers. Got: {args.indices}"
            )
            return
    else:
        samples_to_process_with_indices = initial_samples_with_indices

    if is_mock_mode_via_env and recorded_data_map_for_mocking:
        samples_to_process_with_indices = [
            s_info
            for s_info in samples_to_process_with_indices
            if next(
                (
                    msg["content"]
                    for msg in s_info["data"].get("messages", [])
                    if msg["role"] == "user"
                ),
                None,
            )
            in recorded_data_map_for_mocking
        ]
        if not samples_to_process_with_indices:
            print(
                "Warning: No samples (from current selection) were found in the recorded mock data. No samples will be processed."
            )

    print(
        f"DEBUG: Will attempt to process {len(samples_to_process_with_indices)} samples for this run.\n"
    )

    tasks = []
    semaphore = asyncio.Semaphore(10)  # Consistent concurrency limit

    async with aiohttp.ClientSession() as session:
        for item_info in samples_to_process_with_indices:
            item_data = item_info["data"]
            user_message_content = next(
                (
                    msg["content"]
                    for msg in item_data.get("messages", [])
                    if msg["role"] == "user"
                ),
                None,
            )
            if not user_message_content:
                # This case should ideally be filtered out before task creation
                continue

            tasks.append(
                generate_with_fireworks(
                    semaphore=semaphore,
                    session=session,
                    user_prompt=user_message_content,
                    api_key=api_key,  # type: ignore
                    force_live_api=args.regenerate_recorded_data,
                    recorded_data_map=recorded_data_map_for_mocking,
                )
            )

        valid_samples_for_tasks = [
            s_info
            for s_info in samples_to_process_with_indices
            if next(
                (
                    msg["content"]
                    for msg in s_info["data"].get("messages", [])
                    if msg["role"] == "user"
                ),
                None,
            )
            is not None
        ]

        if tasks:
            regenerated_contents_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        else:
            regenerated_contents_results = []

    actual_processed_count = 0
    error_category_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "API_Error": 0}
    category_c_indices: List[int] = []

    for i, result_or_exc in enumerate(regenerated_contents_results):
        if i >= len(valid_samples_for_tasks):
            print(f"Warning: Mismatch between results and processed samples. Index {i}")
            continue

        item_info = valid_samples_for_tasks[i]
        item = item_info["data"]
        original_sample_index = item_info["index"]
        actual_processed_count += 1
        error_category_for_sample = "N/A"  # For recording

        original_messages_data = item.get("messages")  # This can be None
        user_message_content = ""
        original_assistant_content = None

        if original_messages_data and isinstance(original_messages_data, list):
            # Ensure original_messages_data is not None before iterating
            user_message_content = next(
                (
                    msg.get("content", "")  # Use .get for safety on msg dict
                    for msg in original_messages_data
                    if isinstance(msg, dict) and msg.get("role") == "user"
                ),
                "",
            )
            original_assistant_content = next(
                (
                    msg.get("content")  # Use .get for safety on msg dict
                    for msg in original_messages_data
                    if isinstance(msg, dict) and msg.get("role") == "assistant"
                ),
                None,
            )
        else:  # original_messages_data is None or not a list
            print(
                f"Sample (Original Index {original_sample_index + 1}): Skipping, 'messages' field is missing or not a list in item data."
            )
            all_passed = False
            # Consider this an E type error for categorization purposes if needed, or skip categorization
            continue

        if not original_assistant_content:
            print(
                f"Sample (Original Index {original_sample_index + 1}): Skipping, no original assistant message found for ground truth."
            )
            all_passed = False
            # Consider this an E type error for categorization purposes if needed, or skip categorization
            continue

        print(f"--- Sample (Original Index {original_sample_index + 1}) ---")
        print(f"User Prompt: {user_message_content[:100]}...")

        regenerated_content_str = None
        if isinstance(result_or_exc, Exception):
            print(f"Status: FAILED (API call resulted in exception: {result_or_exc})")
            error_category_counts["API_Error"] += 1
            error_category_for_sample = "API_Error"
            all_passed = False
        else:
            regenerated_content_str = result_or_exc

        if regenerated_content_str is None and not isinstance(result_or_exc, Exception):
            # This case handles if generate_with_fireworks_inner returns None without raising an exception
            # (e.g. API returns no choices, or other non-exception failure)
            print(
                f"Status: FAILED (Could not get/regenerate response from Fireworks API for prompt: {user_message_content[:70]}...)"
            )
            error_category_counts[
                "API_Error"
            ] += 1  # Or a new category like "LLM_No_Response"
            error_category_for_sample = "API_Error"  # Or "LLM_No_Response"
            all_passed = False
            print("---------------------\n")
            if args.regenerate_recorded_data:
                live_call_results_to_record.append(
                    {
                        "user_prompt": user_message_content,
                        "original_assistant_content": original_assistant_content,
                        "regenerated_content": None,
                        "evaluation_score": 0.0,
                        "evaluation_reason": "API/LLM No Response",
                        "error_category": error_category_for_sample,
                    }
                )
            continue

        if regenerated_content_str is not None:  # Only proceed if we have content
            print(f"Regenerated Assistant Response: {regenerated_content_str}")

            messages_for_eval = [
                Message(role="user", content=user_message_content),
                Message(role="assistant", content=regenerated_content_str),
            ]

            try:
                result = math_reward(
                    messages=messages_for_eval,
                    original_messages=messages_for_eval,  # Using regenerated as original for this eval
                    ground_truth=original_assistant_content,
                )

                print(f"Score (for regenerated): {result.score}")
                print(f"Reason (for regenerated): {result.reason}")
                if result.metrics:
                    print("Metrics (for regenerated):")
                    for (
                        metric_name_key,
                        metric_detail,
                    ) in result.metrics.items():
                        print(
                            f"  {metric_name_key}: Score={metric_detail.score}, Success={metric_detail.is_score_valid}, Reason='{metric_detail.reason}'"
                        )

                if result.score == 1.0:
                    print("Status: PASSED")
                    passed_samples += 1
                    error_category_for_sample = "PASSED"
                else:
                    all_passed = False
                    # Programmatic Error Categorization
                    category_determined = False
                    # Category A: Incomplete CoT / Truncated / Empty
                    if regenerated_content_str and (
                        not regenerated_content_str.strip()
                        or "</think>" not in regenerated_content_str
                    ):  # Add check for regenerated_content_str being non-None
                        error_category_counts["A"] += 1
                        error_category_for_sample = "A"
                        category_determined = True
                        print("Status: FAILED (Category A - Incomplete CoT)")

                    if not category_determined:
                        reason_str = (
                            str(result.reason) if result.reason is not None else ""
                        )
                        reason_lower = reason_str.lower()
                        # Category C: Formatting/Extraction Failure
                        if any(
                            err_str in reason_lower
                            for err_str in [
                                "strictness fail",
                                "unboxed 'or'",
                                "ambiguity",
                                "could not extract",
                                "no boxed answer",
                                "malformed",
                            ]
                        ):
                            error_category_counts["C"] += 1
                            error_category_for_sample = "C"
                            category_determined = True
                            print("Status: FAILED (Category C - Formatting/Extraction)")
                        # Category B: Incorrect CoT / Wrong Answer (well-formatted)
                        elif "numeric mismatch" in reason_lower:
                            error_category_counts["B"] += 1
                            error_category_for_sample = "B"
                            category_determined = True
                            print("Status: FAILED (Category B - Numeric Mismatch)")

                    if not category_determined:  # Fallback to E if not A, B, or C
                        error_category_counts["E"] += 1
                        error_category_for_sample = "E"
                        print(f"Status: FAILED (Category E - Other: {result.reason})")

                if error_category_for_sample == "C":
                    category_c_indices.append(original_sample_index)

                if args.regenerate_recorded_data:
                    live_call_results_to_record.append(
                        {
                            "user_prompt": user_message_content,
                            "original_assistant_content": original_assistant_content,
                            "regenerated_content": regenerated_content_str,
                            "evaluation_score": result.score,
                            "evaluation_reason": result.reason,
                            "error_category": error_category_for_sample,
                        }
                    )
                print("---------------------\n")

            except Exception as e:
                print(
                    f"Sample {actual_processed_count}: Error during evaluation of regenerated response - {e}"
                )
                all_passed = False
                error_category_counts["E"] += 1  # Count as Other/Evaluation Error
                error_category_for_sample = "E_EvalError"
                if args.regenerate_recorded_data:
                    live_call_results_to_record.append(
                        {
                            "user_prompt": user_message_content,
                            "original_assistant_content": original_assistant_content,
                            "regenerated_content": regenerated_content_str,
                            "evaluation_score": 0.0,  # Failed due to exception
                            "evaluation_reason": f"Evaluation Exception: {e}",
                            "error_category": error_category_for_sample,
                        }
                    )
                print("---------------------\n")

    print("\n--- Regeneration & Evaluation Summary ---")
    summary_total_processed = actual_processed_count
    print(f"Total samples attempted in this run: {summary_total_processed}")
    print(f"Samples passed (among attempted): {passed_samples}")

    failed_count = 0
    for category_key in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "API_Error",
    ]:  # D is not used yet but kept for structure
        failed_count += error_category_counts.get(category_key, 0)
    # Ensure failed_count is consistent if some samples were skipped before eval
    if (
        summary_total_processed > 0
        and (summary_total_processed - passed_samples) != failed_count
    ):
        # This might happen if a sample was skipped before categorization (e.g. no original_assistant_content)
        # For now, we'll report the sum of categorized errors.
        print(f"Samples failed (sum of categorized errors): {failed_count}")
    else:
        print(f"Samples failed (among attempted): {failed_count}")

    print("\nError Category Counts:")
    for category, count in error_category_counts.items():
        if count > 0:  # Only print categories with errors
            print(f"  Category {category}: {count}")

    if category_c_indices:
        print(
            f"\nIndices of samples with Category C errors: {sorted(list(set(category_c_indices)))}"
        )
    else:
        print("\nNo samples found with Category C errors in this run.")

    if summary_total_processed == 0:
        print(
            "\nNo samples were processed in this run (e.g., no matching mock data found or no samples selected)."
        )
    elif (
        all_passed
    ):  # all_passed would be true if summary_total_processed > 0 and all processed passed
        print(
            "\nAll samples processed in this run passed successfully with regenerated responses!"
        )
    elif summary_total_processed == 0:
        print(
            "\nNo samples were processed in this run (e.g., no matching mock data found)."
        )
    else:
        print(
            "\nSome samples failed or an error occurred during regeneration/evaluation for this run."
        )

    if args.regenerate_recorded_data:
        try:
            with open(recorded_data_path, "w", encoding="utf-8") as f:
                for record in live_call_results_to_record:
                    f.write(json.dumps(record) + "\n")
            print(f"\nSuccessfully saved recorded API calls to {recorded_data_path}")
        except IOError as e:
            print(f"\nError saving recorded API calls to {recorded_data_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate responses for OpenR1 Math Example using Fireworks API and evaluate."  # Modified
    )
    parser.add_argument(
        "--regenerate-recorded-data",
        action="store_true",
        help="Force live API calls and save the prompts, responses, and original truths to a file.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of 0-based dataset indices to process (e.g., '0,5,10'). Processes all if not set.",
    )
    args_parsed = parser.parse_args()  # Renamed to avoid conflict with outer 'args'
    asyncio.run(main(args_parsed))
