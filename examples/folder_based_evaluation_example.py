"""
Example of using the folder-based evaluation interface.
This demonstrates how to preview and deploy evaluations directly from a folder,
with automatic detection of multi-metrics vs single-metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import folder-based evaluation functions
from reward_kit.evaluation import deploy_folder_evaluation, preview_folder_evaluation


def setup_sample_evaluator():
    """Set up a sample evaluator folder structure for demonstration"""
    # Create a root folder for our evaluator
    eval_folder = Path("./sample_evaluator")
    eval_folder.mkdir(exist_ok=True)

    # Create a metrics subfolder
    quality_folder = eval_folder / "quality"
    quality_folder.mkdir(exist_ok=True)

    relevance_folder = eval_folder / "relevance"
    relevance_folder.mkdir(exist_ok=True)

    # Create main.py in the quality folder
    quality_main = quality_folder / "main.py"
    quality_code = '''
def evaluate(messages, original_messages=None, tools=None, **kwargs):
    """
    Evaluate the quality of a response.
    """
    if not messages or len(messages) == 0:
        return {
            "success": False,
            "score": 0.0,
            "reason": "No messages found"
        }

    last_message = messages[-1]
    content = last_message.get("content", "")

    # Simple quality metric based on length and quality markers
    word_count = len(content.split())
    quality_score = min(word_count / 100, 1.0)  # Cap at 1.0

    quality_markers = ["specifically", "example", "detailed", "importantly"]
    marker_count = sum(1 for marker in quality_markers if marker.lower() in content.lower())

    if marker_count > 0:
        quality_score = (quality_score + marker_count / len(quality_markers)) / 2

    return {
        "success": quality_score > 0.6,
        "score": quality_score,
        "reason": f"Quality score based on length ({word_count} words) and quality markers ({marker_count})"
    }
'''
    quality_main.write_text(quality_code)

    # Create main.py in the relevance folder
    relevance_main = relevance_folder / "main.py"
    relevance_code = '''
def evaluate(messages, original_messages=None, tools=None, **kwargs):
    """
    Evaluate the relevance of a response to the original query.
    """
    if not messages or len(messages) == 0:
        return {
            "success": False,
            "score": 0.0,
            "reason": "No messages found"
        }

    # Get the query and the response
    query = ""
    for msg in original_messages or messages[:-1]:
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    last_message = messages[-1]
    response = last_message.get("content", "")

    if not query:
        return {
            "success": False,
            "score": 0.5,
            "reason": "Could not find a user query"
        }

    # Simple relevance check - do words from the query appear in the response?
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())

    # Remove common words
    common_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                   "in", "on", "at", "to", "for", "with", "by", "about", "like"}
    query_words = query_words - common_words

    if not query_words:
        return {
            "success": True,
            "score": 0.7,
            "reason": "Query contained only common words"
        }

    # Calculate overlap
    overlap_words = query_words.intersection(response_words)
    overlap_score = len(overlap_words) / len(query_words)

    return {
        "success": overlap_score > 0.3,
        "score": overlap_score,
        "reason": f"Response includes {len(overlap_words)} of {len(query_words)} meaningful query terms"
    }
'''
    relevance_main.write_text(relevance_code)

    # Create a sample JSONL file
    sample_file = Path("./samples.jsonl")
    samples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the concept of machine learning in detail.",
                },
                {
                    "role": "assistant",
                    "content": "Machine learning is a field of AI that uses algorithms to learn patterns from data. Specifically, it enables computers to improve their performance on tasks through experience. For example, a machine learning model might analyze past customer behavior to predict future purchases.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is deep learning?"},
                {
                    "role": "assistant",
                    "content": "Deep learning is a subset of machine learning that uses neural networks with many layers.",
                },
            ],
            "original_messages": [
                {"role": "user", "content": "What is deep learning?"}
            ],
        },
    ]

    with open(sample_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return eval_folder, sample_file


def clean_up(eval_folder, sample_file):
    """Clean up the created files and folders"""
    # Remove files first
    (eval_folder / "quality" / "main.py").unlink()
    (eval_folder / "relevance" / "main.py").unlink()

    # Remove directories
    (eval_folder / "quality").rmdir()
    (eval_folder / "relevance").rmdir()
    eval_folder.rmdir()

    # Remove sample file
    sample_file.unlink()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Folder-based evaluation example")
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Run in non-interactive mode (skip deploy prompt)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Automatically deploy the evaluator in auto-mode",
    )
    args = parser.parse_args()

    # Set up sample evaluator folder structure
    eval_folder, sample_file = setup_sample_evaluator()

    try:
        # Preview the evaluation
        print("Previewing folder-based evaluation...")
        preview_result = preview_folder_evaluation(
            evaluator_folder=str(eval_folder),
            sample_file=str(sample_file),
            max_samples=2,
        )

        # Display the preview results
        preview_result.display()

        # Check if we should deploy
        should_deploy = False

        if args.auto_mode:
            # In auto mode, check the --deploy flag
            should_deploy = args.deploy
            if should_deploy:
                print("\nAuto-deploying evaluator (--deploy flag set)")
            else:
                print("\nSkipping deployment in auto-mode (--deploy flag not set)")
        else:
            # In interactive mode, ask the user
            deploy_answer = input("\nDo you want to deploy this evaluator? (y/n): ")
            should_deploy = deploy_answer.lower() == "y"

        if should_deploy:
            try:
                print("\nDeploying folder-based evaluation...")
                result = deploy_folder_evaluation(
                    evaluator_id="folder-based-eval",
                    evaluator_folder=str(eval_folder),
                    display_name="Folder-Based Evaluator Example",
                    description="Evaluator with quality and relevance metrics",
                    force=True,  # Overwrite if already exists
                )
                print(
                    f"Successfully deployed evaluator: {result.get('name', 'folder-based-eval')}"
                )
            except Exception as e:
                print(f"Error deploying evaluator: {str(e)}")
                print("Make sure you have proper Fireworks API credentials set up.")
    finally:
        # Clean up
        clean_up(eval_folder, sample_file)


if __name__ == "__main__":
    main()
