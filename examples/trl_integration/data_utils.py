import json
from typing import List, Dict, Any

def process_deepcoder_sample(raw_sample_path: str) -> List[Dict[str, Any]]:
    """
    Reads raw DeepCoder-style samples, extracts relevant information,
    and transforms it into a format suitable for the deepcoder_code_reward function.

    Args:
        raw_sample_path: Path to the JSONL file containing raw samples.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'prompt': The user's problem description (str).
        - 'test_cases': A list of test case dictionaries (list of dicts).
    """
    processed_samples = []
    with open(raw_sample_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                
                # Extract prompt (user content)
                prompt_content = ""
                if isinstance(record.get("prompt"), list) and len(record["prompt"]) > 0:
                    # Assuming the relevant prompt is the first user message
                    for msg in record["prompt"]:
                        if msg.get("role") == "user" and msg.get("content"):
                            prompt_content = msg["content"]
                            break
                if not prompt_content: # Fallback if specific structure not found
                    prompt_content = str(record.get("prompt", ""))
                
                # Extract the target function name for conditional instruction
                target_function = record.get("target_function")

                if target_function:
                    # Instruction for generating a specific function
                    instruction = (
                        f"\n\nIMPORTANT: You are to write a Python function named '{target_function}'. "
                        "Generate ONLY the complete function definition for this function. "
                        "Do not include any example usage, print statements outside the function, "
                        "or any code that reads from stdin or writes to stdout, unless the problem "
                        "description explicitly requires the function itself to perform such I/O."
                    )
                else:
                    # Original instruction for stdin/stdout interaction
                    instruction = (
                        "\n\nIMPORTANT: Your code should read input from standard input (stdin) "
                        "and print the final result to standard output (stdout). "
                        "Only print the final result, nothing else."
                    )
                prompt_content += instruction

                # Parse test cases from ground_truth JSON string
                test_cases_str = record.get("reward_model", {}).get("ground_truth", "[]")
                test_cases = json.loads(test_cases_str)

                # target_function is already extracted above for the instruction
                
                processed_samples.append({
                    "prompt": prompt_content,
                    "test_cases": test_cases,
                    "target_function": target_function
                })
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e} - Line: {line.strip()}")
            except Exception as e:
                print(f"Skipping line due to other error: {e} - Line: {line.strip()}")
                
    return processed_samples

if __name__ == '__main__':
    # Example usage:
    raw_file = "data/simulated_deepcoder_raw_sample.jsonl"
    # Adjust path for direct script execution if necessary
    # This assumes the script is run from examples/trl_integration/
    try:
        transformed_data = process_deepcoder_sample(raw_file)
        for i, item in enumerate(transformed_data):
            print(f"--- Sample {i+1} ---")
            print(f"Prompt: {item['prompt']}")
            print(f"Test Cases: {item['test_cases']}")
            print("-" * 20)
        
        # Example of how to generate the MVP sample dataset (Step 1.C)
        output_mvp_file = "data/deepcoder_mvp_transformed_sample.jsonl"
        with open(output_mvp_file, 'w') as outfile:
            for item in transformed_data:
                outfile.write(json.dumps(item) + '\n')
        print(f"\nTransformed MVP data written to {output_mvp_file}")

    except FileNotFoundError:
        print(f"Error: The file {raw_file} was not found. Make sure the path is correct.")
        print("If running data_utils.py directly, ensure 'data/simulated_deepcoder_raw_sample.jsonl' exists relative to this script's location.")
