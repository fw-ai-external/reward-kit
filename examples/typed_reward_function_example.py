"""
Example showing how to use both the typed_interface and reward_function decorators together.

This demonstrates how to create a deployable reward function that also benefits
from fully typed inputs and outputs during development.
"""

import sys
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import both decorators and typed models
from reward_kit import (
    reward_function, RewardOutput, MetricRewardOutput,   # Original interfaces
    Message, EvaluateResult, MetricResult  # New typed interfaces
)

# Define a reward function using the decorator
@reward_function
def typed_informativeness_reward(
    messages: List[Message],
    **kwargs
) -> EvaluateResult:
    """
    Evaluates how informative a response is using typed interfaces.
    
    This reward function demonstrates how to use both typed code internally
    but still maintain compatibility with the reward_function interface.
    
    Args:
        messages: Conversation messages with full type safety
        **kwargs: Additional arguments
        
    Returns:
        A fully typed evaluation result with metrics
    """
    # If there are no messages, return an error result
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
    
    # Now we can analyze the response with full type safety
    content = last_message.content
    
    # 1. Length check
    word_count = len(content.split())
    length_score = min(word_count / 100.0, 1.0)  # Cap at 100 words
    
    # 2. Specificity markers
    specificity_markers = [
        "specifically", "for example", "such as", "in particular",
        "notably", "precisely", "including", "especially"
    ]
    
    marker_count = sum(1 for marker in specificity_markers if marker.lower() in content.lower())
    marker_score = min(marker_count / 2.0, 1.0)  # Cap at 2 markers
    
    # 3. Structure evaluation
    has_paragraphs = content.count('\n\n') >= 1 or len(content.split('. ')) >= 3
    has_conclusion = any(s.lower().startswith(('in conclusion', 'to summarize', 'therefore')) 
                         for s in content.split('\n'))
    
    structure_score = (float(has_paragraphs) + float(has_conclusion)) / 2.0
    
    # Return metrics with appropriate success status
    return EvaluateResult({
        "length": MetricResult(
            success=length_score > 0.5,
            score=length_score,
            reason=f"Response length: {word_count} words"
        ),
        "specificity": MetricResult(
            success=marker_score > 0.5,
            score=marker_score,
            reason=f"Found {marker_count} specificity markers"
        ),
        "structure": MetricResult(
            success=structure_score > 0.5,
            score=structure_score,
            reason=f"Structure score based on paragraphs ({has_paragraphs}) and conclusion ({has_conclusion})"
        )
    })


def test_reward_function():
    """Test the reward function with sample messages"""
    # Create sample messages - we can use dict-based messages
    sample_messages = [
        {"role": "user", "content": "Explain quantum computing."},
        {"role": "assistant", "content": """
Quantum computing is a type of computation that uses quantum bits or "qubits" instead of classical bits.

Specifically, qubits can exist in multiple states simultaneously due to a property called superposition. This allows quantum computers to process a vast number of possibilities at once. For example, while a 3-bit classical computer can be in only one of 8 possible states at any given time, a 3-qubit quantum computer can represent all 8 states simultaneously.

Another key quantum property is entanglement, which allows qubits to be correlated in ways that classical bits cannot. This enables quantum algorithms to solve certain problems exponentially faster than classical algorithms.

In conclusion, quantum computing represents a fundamentally different approach to computation with the potential to revolutionize fields like cryptography, materials science, and complex system modeling.
"""}
    ]
    
    # Call the function with sample messages
    result = typed_informativeness_reward(messages=sample_messages)
    
    # Print the results
    print("Typed Informativeness Reward Results:\n")
    
    # Calculate aggregate score
    scores = [metric["score"] for metric in result.values()]
    aggregate = sum(scores) / len(scores)
    print(f"Aggregate Score: {aggregate:.2f}")
    
    print("\nIndividual Metrics:")
    for name, metric in result.items():
        print(f"- {name}: {metric['score']:.2f} - {metric['reason']}")
    
    # We can also use properly typed messages if desired
    print("\nDemonstrating with proper typed messages:")
    typed_messages = [
        Message(role="user", content="Explain quantum computing."),
        Message(role="assistant", content="Quantum computing uses quantum bits or qubits.")
    ]
    
    # This will also work, but with full type checking
    typed_result = typed_informativeness_reward(typed_messages)
    print("\nResults with typed messages:")
    for name, metric in typed_result.items():
        print(f"- {name}: {metric['score']:.2f} - {metric['reason']}")
    
    # The reward_function decorator also adds a deploy() method
    print("\nThis function can be deployed with:")
    print("typed_informativeness_reward.deploy(name='typed-informativeness')")


if __name__ == "__main__":
    test_reward_function()