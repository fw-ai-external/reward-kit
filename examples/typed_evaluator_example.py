"""
Example of using the typed interface for an evaluator.
This demonstrates how to use the typed_interface decorator to create
evaluators with full type checking while maintaining backward compatibility.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Literal

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the typed interface and models
from reward_kit import reward_function, Message, EvaluateResult, MetricResult

@reward_function
def evaluate(messages: List[Message], **kwargs) -> EvaluateResult:
    """
    A typed evaluator that measures response quality using different metrics.
    
    Args:
        messages: The conversation messages with full type safety
        **kwargs: Additional arguments
        
    Returns:
        A fully typed evaluation result with metrics
    """
    # If there are no messages, return empty metrics
    if not messages:
        return EvaluateResult({
            "error": MetricResult(success=False, score=0.0, reason="No messages provided")
        })
    
    # Get the last message (assistant's response)
    last_message = messages[-1]
    if last_message.role != "assistant":
        return EvaluateResult({
            "error": MetricResult(success=False, score=0.0, reason="Last message is not from assistant")
        })
    
    content = last_message.content
    
    # Calculate metrics
    # 1. Length metric
    words = content.split()
    word_count = len(words)
    length_score = min(word_count / 100, 1.0)  # Cap at 1.0
    
    # 2. Quality markers
    quality_markers = [
        "specifically", "detailed", "example", "explained", 
        "analysis", "conclusion", "therefore", "importantly"
    ]
    
    marker_count = sum(1 for marker in quality_markers if marker.lower() in content.lower())
    quality_score = min(marker_count / 3.0, 1.0)  # Cap at 1.0 with 3 markers
    
    # 3. Structure metric
    has_introduction = any(s.strip().endswith(('.', ':', '!', '?')) for s in content.split('\n')[:2])
    has_conclusion = any(s.strip().startswith(('In summary', 'To conclude', 'Therefore', 'In conclusion')) 
                         for s in content.split('\n')[-3:])
    has_paragraphs = content.count('\n\n') >= 1
    
    structure_points = sum([has_introduction, has_conclusion, has_paragraphs])
    structure_score = structure_points / 3.0
    
    # Return metrics with appropriate success status
    return EvaluateResult({
        "length": MetricResult(
            success=length_score > 0.5, 
            score=length_score,
            reason=f"Response length: {word_count} words"
        ),
        "quality": MetricResult(
            success=quality_score > 0.5,
            score=quality_score,
            reason=f"Found {marker_count} quality markers"
        ),
        "structure": MetricResult(
            success=structure_score > 0.5,
            score=structure_score,
            reason=f"Structure score based on intro ({has_introduction}), conclusion ({has_conclusion}), and paragraphs ({has_paragraphs})"
        )
    })


def main():
    """Test the typed evaluator with sample data"""
    # Create sample messages
    sample_messages = [
        {"role": "user", "content": "Explain the concept of machine learning."},
        {"role": "assistant", "content": """
Machine learning is a field of artificial intelligence that focuses on developing systems that learn from data.

Specifically, machine learning algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions without being explicitly programmed to do so. For example, a machine learning model might be trained on email messages to learn to distinguish between spam and non-spam.

In conclusion, machine learning represents a fundamental shift in how we approach problem-solving, moving from explicitly programmed solutions to data-driven approaches that can adapt and improve over time.
"""}
    ]
    
    # Call the evaluator with dict-based messages
    result = evaluate(sample_messages)
    print("Evaluation Results:")
    print(json.dumps(result, indent=2))
    print()
    
    # Calculate aggregate score
    scores = [metric["score"] for metric in result.values()]
    aggregate = sum(scores) / len(scores)
    print(f"Aggregate Score: {aggregate:.2f}")
    
    # Show that we can also call with properly typed messages if desired
    from reward_kit import Message
    typed_messages = [
        Message(role="user", content="Explain quantum computing."),
        Message(role="assistant", content="Quantum computing uses quantum bits or qubits.")
    ]
    
    # This will also work, but with full type checking
    typed_result = evaluate(typed_messages)  # type checker knows this is List[Message]
    print("\nTyped Messages Result:")
    print(json.dumps(typed_result, indent=2))


if __name__ == "__main__":
    main()