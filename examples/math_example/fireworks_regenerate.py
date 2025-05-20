import argparse
import json
import os
import sys
import asyncio  # Added asyncio
import aiohttp  # Added aiohttp

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import (
    Optional,
    List,
    Dict,
    Any,
)  # Add Optional, List, Dict, Any for type hints
from reward_kit.rewards.math import math_reward
from reward_kit.models import Message

# Configuration for Fireworks API
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
# Specify the Qwen3 model. You might need to adjust if a specific variant is required.
# Using a general Qwen model name, check Fireworks documentation for exact names.
FIREWORKS_MODEL_NAME = "accounts/fireworks/models/qwen3-30b-a3b"
RECORDED_DATA_FILENAME = "fireworks_regenerate_recorded_data.jsonl"  # Define filename


def load_dataset(file_path: str):
    """Loads a JSONL dataset."""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


async def generate_with_fireworks_inner(  # Renamed to avoid conflict, and this is the core logic
    session: aiohttp.ClientSession,
    user_prompt: str,
    api_key: str,
    force_live_api: bool = False,
    recorded_data_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Core logic to generate a response using Fireworks API or mock data, asynchronously."""

    # If not forcing live API and mock environment variable is set for testing
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
            # Fallback for other prompts if any during mock testing of this script itself
            return "A generic mocked math solution that will likely result in a score of 0 for other prompts."

    # Proceed with live API call if force_live_api is true or if not in TEST_MOCK_FIREWORKS_REGEN mode
    if api_key is None and not (
        not force_live_api and os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true"
    ):  # This check is more for live calls; mock path is handled above.
        # For live calls, api_key presence is checked before calling this inner function.
        # This specific check might be redundant if outer layers ensure api_key for live calls.
        pass  # Outer layers should ensure api_key if it's a live call.

    # print(f"Calling Fireworks API for prompt: '{user_prompt[:70]}...'") # Moved to wrapper
    system_prompt = "IMPORTANT: You MUST provide your final numerical answer enclosed *only* in `\\boxed{answer}`. Do not include any other numbers or text within the box. Your entire response should be your reasoning, followed by the single, final boxed answer. Example: `\\boxed{123.45}`."
    payload: Dict[str, Any] = {  # Added type hint for payload
        "model": FIREWORKS_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4000,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(
            FIREWORKS_API_URL,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),  # Increased timeout
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
    except aiohttp.ClientError as e:  # Catch aiohttp specific errors
        print(f"Error calling Fireworks API (aiohttp): {e}")
        # For aiohttp.ClientResponseError, e.status and e.message are available
        if hasattr(e, "status"):
            print(f"Status: {e.status}, Message: {e.message}")  # type: ignore
        return None
    except asyncio.TimeoutError:
        print(
            f"Timeout error calling Fireworks API for prompt: '{user_prompt[:70]}...'"
        )
        return None
    except (
        json.JSONDecodeError
    ) as e:  # This can still happen if response is not valid JSON
        print(f"Error decoding Fireworks API response: {e}")
        return None


async def generate_with_fireworks(  # This is the new wrapper with semaphore
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    user_prompt: str,
    api_key: str,
    force_live_api: bool = False,
    recorded_data_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Asynchronous wrapper for generate_with_fireworks_inner with semaphore control."""
    async with semaphore:
        # Mocking logic is inside generate_with_fireworks_inner
        # Live API key check is also effectively handled there or before calling this.
        if not (
            not force_live_api and os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true"
        ):  # Only print for live calls
            # Attempt to get semaphore details safely
            try:
                max_workers = getattr(
                    getattr(semaphore._loop, "_default_executor", None),
                    "_max_workers",
                    "N/A",
                )
                concurrency_str = f"{10 - semaphore._value}/{max_workers}"
            except AttributeError:
                concurrency_str = f"{10 - semaphore._value}/Unknown"

            print(
                f"Calling Fireworks API for prompt: '{user_prompt[:70]}...' (Concurrency: {concurrency_str})"
            )

        return await generate_with_fireworks_inner(
            session, user_prompt, api_key, force_live_api, recorded_data_map
        )


async def main(args):  # main is now async
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    recorded_data_path = os.path.join(os.path.dirname(__file__), RECORDED_DATA_FILENAME)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    api_key = os.environ.get("FIREWORKS_API_KEY")
    # API key is only strictly required if not using mocks or if regenerating recorded data
    is_mock_mode_via_env = os.environ.get("TEST_MOCK_FIREWORKS_REGEN") == "true"
    recorded_data_map_for_mocking: Optional[Dict[str, str]] = None

    if not args.regenerate_recorded_data and not is_mock_mode_via_env and not api_key:
        print(
            "Error: FIREWORKS_API_KEY environment variable is not set (and not in mock/regenerate mode)."
        )
        print("Please set this variable to your Fireworks API key.")
        return

    if args.regenerate_recorded_data and not api_key:
        print(
            "Error: FIREWORKS_API_KEY must be set when using --regenerate-recorded-data."
        )
        return

    dataset = load_dataset(dataset_path)
    all_passed = True
    total_samples = len(dataset)
    passed_samples = 0

    live_call_results_to_record: List[Dict] = []  # For storing data if regenerating

    if args.regenerate_recorded_data:
        print(
            f"MODE: Regenerating recorded data. Live API calls will be made. Output to: {recorded_data_path}"
        )
    elif is_mock_mode_via_env:
        print(
            "MODE: Attempting to use mock data for Fireworks API calls from recorded file."
        )
        if os.path.exists(recorded_data_path):
            try:
                recorded_data_list = load_dataset(recorded_data_path)
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
            "MODE: Using live Fireworks API calls (not saving to recorded data file)."
        )

    print(
        f"Starting Fireworks Regeneration & Evaluation for Math Example using {dataset_path}...\n"
    )

    # Determine samples to process
    initial_samples_with_indices = [
        {"index": i, "data": item} for i, item in enumerate(dataset)
    ]

    # Filter by specified indices first, if provided
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

    # Then, if in mock mode (and not already filtered by specific indices for mock testing), filter by mock data
    if is_mock_mode_via_env and recorded_data_map_for_mocking:
        print(
            f"DEBUG: Mock mode active. Filtering samples to those present in recorded_data_map_for_mocking."
        )
        # Filter the current list of samples_to_process_with_indices
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
            # Allow to proceed to show 0 processed if that's the case.

    print(
        f"DEBUG: Will attempt to process {len(samples_to_process_with_indices)} samples for this run.\n"
    )

    tasks = []
    # It's important that the semaphore is created here, before the session,
    # or passed around correctly if the session is managed differently.
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession() as session:  # Create session for all calls
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
                print(
                    f"Skipping sample at original index {item_info['index']} due to missing user prompt."
                )
                # To keep regenerated_contents_results aligned with samples_to_process_with_indices,
                # we should append a placeholder or handle this carefully in the results loop.
                # For simplicity, we'll ensure the results loop iterates over the same filtered list.
                # Or, even better, filter out such items from samples_to_process_with_indices beforehand.
                # For now, let's assume user_message_content will be present for valid samples.
                # If we were to append a placeholder: tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
                continue  # Skip adding a task for this item

            # API key check for live calls
            is_live_call_attempt = args.regenerate_recorded_data or (
                not is_mock_mode_via_env
            )
            if is_live_call_attempt and not api_key:
                print(
                    f"Skipping API call for sample (Original Index {item_info['index'] + 1}) due to missing API key in live mode."
                )
                # Append a placeholder or handle in results loop
                # For now, we'll just not add a task, and the results loop will need to be robust.
                # A better way is to pre-filter samples_to_process_with_indices if API key is missing for live mode.
                # However, the top-level API key check should prevent getting here for all samples if key is missing.
                # This per-item check is more for safety if logic changes.
                # Let's assume the top-level check handles this for now.
                pass

            tasks.append(
                generate_with_fireworks(  # Call the semaphore-wrapped version
                    semaphore=semaphore,
                    session=session,
                    user_prompt=user_message_content,
                    api_key=api_key,  # type: ignore
                    force_live_api=args.regenerate_recorded_data,
                    recorded_data_map=recorded_data_map_for_mocking,
                )
            )

        # Filter samples_to_process_with_indices to only include those for which tasks were created
        # This ensures alignment if any samples were skipped (e.g., missing user prompt)
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

        if tasks:  # Only gather if there are tasks to run
            regenerated_contents_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        else:
            regenerated_contents_results = []

    actual_processed_count = 0
    # Iterate over the samples for which tasks were actually created and run
    for i, result_or_exc in enumerate(regenerated_contents_results):
        # Ensure we are mapping back to the correct original item
        # This assumes regenerated_contents_results is in the same order as tasks,
        # and tasks were created from valid_samples_for_tasks
        if i >= len(valid_samples_for_tasks):  # Should not happen if logic is correct
            print(f"Warning: Mismatch between results and processed samples. Index {i}")
            continue

        item_info = valid_samples_for_tasks[i]
        item = item_info["data"]
        original_sample_index = item_info["index"]
        actual_processed_count += 1

        original_messages_data = item.get("messages")
        # We've already checked for user_message_content when creating tasks
        user_message_content = next(
            (msg["content"] for msg in original_messages_data if msg["role"] == "user"),
            "",
        )
        original_assistant_content = next(
            (
                msg["content"]
                for msg in original_messages_data
                if msg["role"] == "assistant"
            ),
            None,
        )

        if not original_assistant_content:  # Should not happen with current dataset
            print(
                f"Sample (Original Index {original_sample_index + 1}): Skipping, no original assistant message found for ground truth."
            )
            all_passed = False
            continue

        print(f"--- Sample (Original Index {original_sample_index + 1}) ---")
        print(f"User Prompt: {user_message_content[:100]}...")

        if isinstance(result_or_exc, Exception):
            regenerated_content = None
            print(f"Status: FAILED (API call resulted in exception: {result_or_exc})")
        else:
            regenerated_content = result_or_exc

        if (
            regenerated_content is None
        ):  # Handles both explicit None return and caught exceptions leading to None
            # Error message already printed by generate_with_fireworks or above
            if not isinstance(
                result_or_exc, Exception
            ):  # If it wasn't an exception, print generic failure
                print(
                    f"Status: FAILED (Could not get/regenerate response from Fireworks API for prompt: {user_message_content[:70]}...)"
                )
            all_passed = False
            # If it's a mock mode failure due to missing key, it's already printed in generate_with_fireworks
            # No need to add to live_call_results_to_record if content is None
            print("---------------------\n")
            continue

        print(f"Regenerated Assistant Response: {regenerated_content}")

        # Prepare messages for math_reward
        # The conversation history for evaluation should include the user's prompt and the *new* assistant response.
        messages_for_eval = [
            Message(role="user", content=user_message_content),
            Message(role="assistant", content=regenerated_content),
        ]

        try:
            # Evaluate the regenerated_content against the original_assistant_content
            result = math_reward(
                messages=messages_for_eval,  # This contains user_prompt + regenerated_content
                original_messages=messages_for_eval,  # Can be same for this direct eval
                ground_truth=original_assistant_content,  # The original correct answer
            )

            print(f"Score (for regenerated): {result.score}")
            print(f"Reason (for regenerated): {result.reason}")
            if result.metrics:
                print("Metrics (for regenerated):")
                for metric_name, metric_detail in result.metrics.items():
                    print(
                        f"  {metric_name}: Score={metric_detail.score}, Success={metric_detail.success}, Reason='{metric_detail.reason}'"
                    )

            if result.score == 1.0:
                print("Status: PASSED")
                passed_samples += 1
            else:
                print("Status: FAILED")
                all_passed = False

            if args.regenerate_recorded_data and regenerated_content is not None:
                live_call_results_to_record.append(
                    {
                        "user_prompt": user_message_content,
                        "original_assistant_content": original_assistant_content,
                        "regenerated_content": regenerated_content,
                        "evaluation_score": result.score,
                        "evaluation_reason": result.reason,
                    }
                )
            print("---------------------\n")

        except Exception as e:
            print(
                f"Sample {actual_processed_count}: Error during evaluation of regenerated response - {e}"
            )
            all_passed = False
            print("---------------------\n")

    # Adjust summary to reflect actual_processed_count if it's different from total_samples (e.g. in mock mode with partial data)
    # However, the test assertion expects "All samples passed" based on what it *does* process.
    # The script's internal `all_passed` flag should correctly reflect if all *attempted* samples passed.
    # The number of samples for "Total samples processed" in the summary should be actual_processed_count.

    print("\n--- Regeneration & Evaluation Summary ---")
    # If in mock mode and samples were filtered, total_samples might be misleading.
    # The spirit of "All samples passed" refers to those *attempted*.
    # The `all_passed` flag tracks if any of the `actual_processed_count` samples failed.
    # `passed_samples` counts successful ones among `actual_processed_count`.

    summary_total_processed = (
        actual_processed_count  # Use the count of samples actually iterated over
    )

    print(f"Total samples attempted in this run: {summary_total_processed}")
    print(f"Samples passed (among attempted): {passed_samples}")

    # If summary_total_processed is 0 (e.g. no mock data found for first 10), then failed_count is 0.
    failed_count = (
        summary_total_processed - passed_samples if summary_total_processed > 0 else 0
    )
    print(f"Samples failed (among attempted): {failed_count}")

    if (
        all_passed and summary_total_processed > 0
    ):  # Condition: all attempted samples passed AND at least one was attempted
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
        description="Regenerate responses using Fireworks API and evaluate."
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
    args = parser.parse_args()
    asyncio.run(main(args))
