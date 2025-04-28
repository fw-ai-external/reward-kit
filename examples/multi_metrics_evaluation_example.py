"""
Example of creating a multi-metrics evaluation.
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
    tmp_folder = Path("./tmp_multi_metric")
    tmp_folder.mkdir(exist_ok=True)
    
    # Create a main.py file with an evaluate function for multiple metrics
    main_py = tmp_folder / "main.py"
    
    # Using single quotes for outer string to avoid conflict with inner triple quotes
    evaluate_code = '''
def evaluate(messages, original_messages=None, tools=None, **kwargs):
    """
    Evaluate a sample entry with multiple metrics.
    Returns multiple metrics in a single evaluation.
    """
    # If this is the first message, there's nothing to evaluate
    if not messages:
        return {
            'score': 0.0, 
            'reason': 'No messages found',
            'metrics': {}
        }
    
    # Get the last message (assistant's response)
    last_message = messages[-1]
    content = last_message.get('content', '')
    
    # Word count metric
    word_count = len(content.split())
    word_count_score = min(word_count / 100, 1.0)  # Cap at 1.0
    
    # Character count metric
    char_count = len(content)
    char_count_score = min(char_count / 1000, 1.0)  # Cap at 1.0
    
    # Tool usage metric if tools were available
    tool_usage_score = 0.5  # Default
    if tools:
        # Higher score if tools were available and used
        tool_usage = kwargs.get('tool_usage', False)
        tool_usage_score = 1.0 if tool_usage else 0.0
    
    # Calculated a weighted average score
    final_score = 0.5 * word_count_score + 0.3 * char_count_score + 0.2 * tool_usage_score
    
    return {
        'score': final_score,
        'reason': f'Combined score based on multiple metrics',
        'metrics': {
            'word_count': {
                'score': word_count_score,
                'reason': f'Word count: {word_count}'
            },
            'char_count': {
                'score': char_count_score,
                'reason': f'Character count: {char_count}'
            },
            'tool_usage': {
                'score': tool_usage_score,
                'reason': f'Tool usage score based on available tools'
            }
        }
    }
'''
    
    main_py.write_text(evaluate_code)
    
    # Create a utility file to show multiple file support
    utils_py = tmp_folder / "utils.py"
    utils_code = '''
def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def count_chars(text):
    """Count the number of characters in a text."""
    return len(text)

def calculate_tool_score(tools, used=False):
    """Calculate a score based on tool usage."""
    if not tools:
        return 0.0
    return 1.0 if used else 0.0
'''
    utils_py.write_text(utils_code)
    
    # Create a sample JSONL file
    sample_file = Path("./multi_samples.jsonl")
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
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information"
                    }
                }
            ],
            "tool_usage": True
        }
    ]
    
    with open(sample_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    # Preview the evaluation with multi-metrics
    print("Previewing multi-metrics evaluation...")
    preview_result = preview_evaluation(
        multi_metrics=True,
        folder="./tmp_multi_metric",
        sample_file="./multi_samples.jsonl",
        max_samples=2
    )
    
    preview_result.display()
    
    # Create the evaluation
    print("\nCreating multi-metrics evaluation...")
    try:
        evaluator = create_evaluation(
            evaluator_id="multi-metrics-eval",
            multi_metrics=True,
            folder="./tmp_multi_metric",
            display_name="Multi-Metrics Evaluator",
            description="Evaluates responses based on multiple metrics (word count, character count, tool usage)"
        )
        print(f"Created evaluator: {evaluator['name']}")
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
        print("Make sure you have proper Fireworks API credentials set up.")
    
    # Clean up
    main_py.unlink()
    utils_py.unlink()
    tmp_folder.rmdir()
    sample_file.unlink()

if __name__ == "__main__":
    main()