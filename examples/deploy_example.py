"""
Example of deploying a reward function to Fireworks.

This example demonstrates how to create and deploy a reward function
that evaluates the informativeness of an assistant's response.
"""

import os
from typing import List, Dict, Optional
from reward_kit import reward_function, RewardOutput, MetricRewardOutput

@reward_function
def informativeness_reward(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """
    Evaluates the informativeness of an assistant response based on 
    specificity markers and content density.
    """
    # Get the assistant's response
    if not messages or messages[-1].get("role") != "assistant":
        return RewardOutput(score=0.0, metrics={
            "error": MetricRewardOutput(score=0.0, reason="No assistant response found")
        })
    
    response = messages[-1].get("content", "")
    metrics = {}
    
    # 1. Length check - reward concise but informative responses
    length = len(response)
    length_score = min(length / 1000.0, 1.0)  # Cap at 1000 chars
    metrics["length"] = MetricRewardOutput(
        score=length_score * 0.2,  # 20% weight
        reason=f"Response length: {length} chars"
    )
    
    # 2. Specificity markers
    specificity_markers = [
        "specifically", "in particular", "for example", 
        "such as", "notably", "precisely", "exactly"
    ]
    marker_count = sum(1 for marker in specificity_markers if marker.lower() in response.lower())
    marker_score = min(marker_count / 2.0, 1.0)  # Cap at 2 markers
    metrics["specificity"] = MetricRewardOutput(
        score=marker_score * 0.3,  # 30% weight
        reason=f"Found {marker_count} specificity markers"
    )
    
    # 3. Content density (simple heuristic based on ratio of content words to total)
    content_words = ['information', 'data', 'analysis', 'recommend', 
                     'solution', 'approach', 'technique', 'method']
    word_count = len(response.split())
    content_word_count = sum(1 for word in content_words if word.lower() in response.lower())
    
    if word_count > 0:
        density_score = min(content_word_count / (word_count / 20), 1.0)  # Normalize by expecting ~5% density
    else:
        density_score = 0.0
        
    metrics["content_density"] = MetricRewardOutput(
        score=density_score * 0.5,  # 50% weight
        reason=f"Content density: {content_word_count} content words in {word_count} total words"
    )
    
    # Calculate final score as weighted sum of metrics
    final_score = sum(metric.score for metric in metrics.values())
    
    return RewardOutput(score=final_score, metrics=metrics)

# Test the reward function with example messages
def test_reward_function():
    # Example messages
    test_messages = [
        {"role": "user", "content": "Can you explain machine learning?"},
        {"role": "assistant", "content": "Machine learning is a method of data analysis that automates analytical model building. Specifically, it uses algorithms that iteratively learn from data, allowing computers to find hidden insights without being explicitly programmed where to look. For example, deep learning is a type of machine learning that uses neural networks with many layers. Such approaches have revolutionized fields like computer vision and natural language processing."}
    ]
    
    # Test the reward function
    result = informativeness_reward(messages=test_messages, original_messages=[test_messages[0]])
    print("Informativeness Reward Result:")
    print(f"Score: {result.score}")
    print("Metrics:")
    for name, metric in result.metrics.items():
        print(f"  {name}: {metric.score} - {metric.reason}")
    print()
    
    return result

# Deploy the reward function to Fireworks
def deploy_to_fireworks():
    # Read settings file to get account_id
    import configparser
    from pathlib import Path
    
    # First check environment variables
    api_base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    is_dev = "dev.api.fireworks.ai" in api_base
    
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    auth_token = os.environ.get("FIREWORKS_API_KEY")
    
    if account_id:
        print(f"Using account ID from environment: {account_id}")
    
    if auth_token:
        print(f"Using auth token from environment")
        print(f"Token starts with: {auth_token[:10]}...")
    
    # If not in environment, try config files
    try:
        # Only get account_id from settings if not already set
        if not account_id:
            settings_path = Path.home() / ".fireworks" / "settings.ini"
            if settings_path.exists():
                # For settings.ini, we'll manually parse it since we know the format
                with open(settings_path, 'r') as f:
                    for line in f:
                        if "account_id" in line and "=" in line:
                            account_id = line.split("=")[1].strip()
                            break
                            
                if account_id:
                    print(f"Using account ID from settings: {account_id}")
                else:
                    account_id = "pyroworks-dev"  # Default value
                    print(f"No account_id found in settings.ini, using default: {account_id}")
            else:
                print("No settings.ini file found")
                account_id = "pyroworks-dev"  # Default value
        
        # Only get auth token if not already set
        if not auth_token:
            auth_path = Path.home() / ".fireworks" / "auth.ini"
            if auth_path.exists():
                # For auth.ini, we'll manually parse it since we know the format
                with open(auth_path, 'r') as f:
                    for line in f:
                        # Look for the appropriate token based on environment
                        key_name = "api_key"
                        if key_name in line and "=" in line:
                            auth_token = line.split("=")[1].strip()
                            break
                
                if auth_token:
                    print(f"Found auth token for {'dev' if is_dev else 'prod'} in auth.ini")
                    print(f"Token starts with: {auth_token[:10]}...")
                else:
                    print(f"No {key_name} found in auth.ini")
    except Exception as e:
        print(f"Error reading config: {str(e)}")
        if not account_id:
            account_id = "pyroworks-dev"  # Default value
        # Don't set a default for auth_token
        
    # Deploy the reward function
    evaluation_id = informativeness_reward.deploy(
        name="informativeness-v1",
        description="Evaluates response informativeness based on specificity and content density",
        account_id=account_id,
        auth_token=auth_token
    )
    print(f"Deployed evaluation with ID: {evaluation_id}")
    
    # Example of deploying with a custom provider
    custom_evaluation_id = informativeness_reward.deploy(
        name="informativeness-v1-anthropic",
        description="Informativeness evaluation using Claude model",
        account_id=account_id,
        auth_token=auth_token,
        providers=[
            {
                "providerType": "anthropic",
                "modelId": "claude-3-sonnet-20240229"
            }
        ]
    )
    print(f"Deployed evaluation with custom provider: {custom_evaluation_id}")
    
    # Show how to use the evaluation ID in a training job
    print("Use this in your RL training job:")
    print(f"firectl create rl-job --reward-endpoint \"https://api.fireworks.ai/v1/evaluations/{evaluation_id}\"")
    
    return evaluation_id

if __name__ == "__main__":
    # First test the reward function locally
    print("Testing reward function locally...")
    test_reward_function()
    
    # Deploy to Fireworks
    print("\nDeploying to Fireworks...")
    deploy_to_fireworks()