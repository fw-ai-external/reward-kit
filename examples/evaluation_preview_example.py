"""
Example of previewing an evaluation before creation.
"""

import os
import sys
import json
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# No example mode - will use real authentication

from reward_kit.evaluation import preview_evaluation, create_evaluation

def main():
    # Create a temporary example folder
    tmp_folder = Path("./tmp_metric")
    tmp_folder.mkdir(exist_ok=True)
    
    # Create a main.py file with an evaluate function
    main_py = tmp_folder / "main.py"
    
    # Using single quotes for outer string to avoid conflict with inner triple quotes
    evaluate_code = '''
def evaluate(messages, original_messages=None, tools=None, **kwargs):
    """
    Evaluate a sample entry.
    
    Args:
        messages: List of conversation messages
        original_messages: Original messages (usually without the response being evaluated)
        tools: Available tools for the conversation
        **kwargs: Additional parameters
        
    Returns:
        Dict with score and metrics information
    """
    # If this is the first message, there's nothing to evaluate
    if not messages:
        return {'score': 0.0, 'reason': 'No messages found'}
    
    # Get the last message (assistant's response)
    last_message = messages[-1]
    content = last_message.get('content', '')
    
    # Simple evaluation: count the number of words
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)  # Cap at 1.0
    
    return {
        'score': score,
        'reason': f'Word count: {word_count}',
        'metrics': {
            'word_count': {
                'score': score,
                'reason': f'Response has {word_count} words'
            }
        }
    }
'''
    
    main_py.write_text(evaluate_code)
    
    # Create a sample JSONL file
    sample_file = Path("./samples.jsonl")
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Tell me about AI"},
                {"role": "assistant", "content": "AI (Artificial Intelligence) refers to systems designed to mimic human intelligence. These systems can learn from data, identify patterns, and make decisions with minimal human intervention."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on building systems that can learn from and make decisions based on data."}
            ],
            "original_messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on building systems that can learn from and make decisions based on data."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information"
                    }
                }
            ]
        }
    ]
    
    with open(sample_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    # Preview the evaluation
    print("Previewing evaluation...")
    preview_result = preview_evaluation(
        metric_folders=["word_count=./tmp_metric"],
        sample_file="./samples.jsonl",
        max_samples=2
    )
    
    preview_result.display()
    
    # Modified approach - add a flag to reward_kit.evaluation that we'll check
    # to determine if the preview API was successfully used
    import reward_kit.evaluation as evaluation_module
    
    # Check if 'used_preview_api' attribute exists and is True
    # This attribute would be set to True when the preview API is used
    # and False when fallback mode is used
    if hasattr(evaluation_module, 'used_preview_api') and not evaluation_module.used_preview_api:
        print("Note: The preview used fallback mode due to server issues.")
        proceed = input("The server might be having connectivity issues. Do you want to try creating the evaluator anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Skipping evaluator creation.")
            # Clean up and exit
            main_py.unlink()
            tmp_folder.rmdir()
            sample_file.unlink()
            sys.exit(0)
    
    print("\nCreating evaluation...")
    try:
        evaluator = create_evaluation(
            evaluator_id="word-count-eval",
            metric_folders=["word_count=./tmp_metric"],
            display_name="Word Count Evaluator",
            description="Evaluates responses based on word count",
            force=True  # Update the evaluator if it already exists
        )
        print(f"Created evaluator: {evaluator['name']}")
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        print("Make sure you have proper Fireworks API credentials set up.")
    
    # Clean up
    main_py.unlink()
    tmp_folder.rmdir()
    sample_file.unlink()

if __name__ == "__main__":
    main()