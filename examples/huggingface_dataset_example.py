"""
Example demonstrating integration with HuggingFace datasets for evaluation preview.
"""

import os
import sys
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check for required environment variables
if not os.environ.get("FIREWORKS_API_KEY"):
    print("Warning: FIREWORKS_API_KEY environment variable is not set.")
    print("Either set this variable or provide an auth_token when calling create_evaluation().")
    print("Example: FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY python examples/huggingface_dataset_example.py")

# Import the evaluation functions
from reward_kit.evaluation import preview_evaluation, huggingface_dataset_to_jsonl
from reward_kit.rewards.lean_prover import lean_prover_reward, deepseek_prover_v2_reward

def main():
    # Example 1: Convert a HuggingFace dataset to JSONL for manual inspection
    print("Converting DeepSeek-ProverBench dataset to JSONL...")
    jsonl_file = huggingface_dataset_to_jsonl(
        dataset_name="deepseek-ai/DeepSeek-ProverBench",
        split="test",
        max_samples=5,
        prompt_key="statement",
        response_key="reference_solution"
    )
    print(f"Dataset converted to JSONL file: {jsonl_file}")
    
    # Example 2: Preview evaluation using HuggingFace dataset directly
    print("\nPreviewing evaluation with DeepSeek-ProverBench dataset...")
    
    # Create a simple reward function metric folder
    os.makedirs("./temp_metrics/deepseek_prover", exist_ok=True)
    
    # Create main.py with deepseek_prover_v2_reward
    with open("./temp_metrics/deepseek_prover/main.py", "w") as f:
        f.write("""
from reward_kit.rewards.lean_prover import deepseek_prover_v2_reward

def evaluate(messages, original_messages=None, tools=None, **kwargs):
    # Extract user prompt (statement) and assistant response (proof)
    statement = messages[0]["content"] if messages and len(messages) > 0 else ""
    proof = messages[1]["content"] if messages and len(messages) > 1 else ""
    
    # Use the statement as problem statement and evaluate the proof
    result = deepseek_prover_v2_reward(
        response=proof,
        statement=statement,
        check_subgoals=True,
        verbose=True
    )
    
    return {
        "score": result.score,
        "reasoning": f"Proof quality: {result.score:.2f}"
    }
""")
    
    # Preview the evaluation
    preview_result = preview_evaluation(
        metric_folders=["deepseek_prover=./temp_metrics/deepseek_prover"],
        huggingface_dataset="deepseek-ai/DeepSeek-ProverBench",
        huggingface_split="test",
        max_samples=3,
        huggingface_prompt_key="statement",
        huggingface_response_key="reference_solution"
    )
    
    # Display results
    preview_result.display()
    
    # Clean up temporary files
    import shutil
    shutil.rmtree("./temp_metrics", ignore_errors=True)
    
    # Example 3: Evaluate a custom response against the DeepSeek-ProverBench dataset
    print("\nEvaluating a custom response against DeepSeek-ProverBench statement...")
    
    # Get a statement from the dataset for demonstration
    try:
        from datasets import load_dataset
        dataset = load_dataset("deepseek-ai/DeepSeek-ProverBench", split="test")
        statement = dataset[0]["statement"]
        
        # Sample custom proof (simplified for illustration)
        custom_proof = """
theorem my_theorem : 
  ∀ n : ℕ, n + 0 = n :=
begin
  intro n,
  rw add_zero,
  refl,
end
"""
        
        # Evaluate the proof
        result = deepseek_prover_v2_reward(
            response=custom_proof,
            statement=statement,
            verbose=True
        )
        
        print(f"Statement: {statement}")
        print(f"Custom proof score: {result.score}")
        if result.metrics:
            print("Detailed metrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"  {metric_name}: {metric_value.score} - {metric_value.reason}")
    
    except ImportError:
        print("Could not load datasets package. Install with: pip install 'reward-kit[deepseek]'")

if __name__ == "__main__":
    main()