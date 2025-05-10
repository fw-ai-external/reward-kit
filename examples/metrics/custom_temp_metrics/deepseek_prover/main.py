
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from examples.basic_reward import combined_reward

def evaluate(messages, original_messages=None, tools=None, **kwargs):
    # Evaluate the messages using the combined_reward function
    result = combined_reward(
        messages=messages,
        **kwargs
    )
    
    return {
        "score": result['score'],
        "reasoning": result['reason'],
        "metrics": result['metrics']
    }
