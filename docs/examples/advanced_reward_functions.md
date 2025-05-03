# Advanced Reward Functions

This guide demonstrates how to create advanced reward functions with multiple metrics and complex evaluation logic.

## Multi-Component Reward Function

This example shows a comprehensive reward function that evaluates multiple aspects of a response.

```python
"""
Example of an advanced multi-component reward function.
"""

from typing import List, Dict, Optional, Any
from reward_kit import reward_function, RewardOutput, MetricRewardOutput
import re

@reward_function
def comprehensive_evaluator(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RewardOutput:
    """
    Comprehensive reward function that evaluates multiple aspects
    of a response: helpfulness, accuracy, safety, and conciseness.
    
    Args:
        messages: List of conversation messages
        original_messages: Original context messages
        metadata: Optional configuration parameters
        **kwargs: Additional parameters
        
    Returns:
        RewardOutput with score and metrics
    """
    # Default settings
    metadata = metadata or {}
    weights = metadata.get("weights", {
        "helpfulness": 0.3,
        "accuracy": 0.3,
        "safety": 0.3,
        "conciseness": 0.1
    })
    
    # Get user query and assistant response
    if not messages or len(messages) < 2:
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="Insufficient messages")}
        )
    
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
            
    assistant_response = messages[-1].get("content", "")
    if not assistant_response or messages[-1].get("role") != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="No assistant response found")}
        )
    
    # Initialize metrics dictionary
    metrics = {}
    
    # 1. Helpfulness evaluation
    helpfulness_score, helpfulness_reason = evaluate_helpfulness(user_query, assistant_response)
    metrics["helpfulness"] = MetricRewardOutput(
        score=helpfulness_score,
        reason=helpfulness_reason
    )
    
    # 2. Accuracy evaluation (simplified example)
    accuracy_score, accuracy_reason = evaluate_accuracy(assistant_response)
    metrics["accuracy"] = MetricRewardOutput(
        score=accuracy_score,
        reason=accuracy_reason
    )
    
    # 3. Safety evaluation
    safety_score, safety_reason = evaluate_safety(assistant_response)
    metrics["safety"] = MetricRewardOutput(
        score=safety_score,
        reason=safety_reason
    )
    
    # 4. Conciseness evaluation
    conciseness_score, conciseness_reason = evaluate_conciseness(assistant_response)
    metrics["conciseness"] = MetricRewardOutput(
        score=conciseness_score,
        reason=conciseness_reason
    )
    
    # Calculate final score using weighted sum
    final_score = sum(
        metrics[key].score * weights.get(key, 0.0)
        for key in metrics
        if key in weights
    )
    
    # Add metadata-based adjustments if specified
    if "boost_factor" in metadata:
        boost = float(metadata["boost_factor"])
        final_score = min(final_score * boost, 1.0)
        metrics["boost_applied"] = MetricRewardOutput(
            score=0.0,  # Doesn't affect score calculation
            reason=f"Applied boost factor of {boost}"
        )
    
    return RewardOutput(score=final_score, metrics=metrics)


# Helper evaluation functions
def evaluate_helpfulness(query: str, response: str) -> tuple[float, str]:
    """Evaluate how helpful the response is for the query."""
    # Check if response directly addresses the query keywords
    query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))
    response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
    
    # Calculate keyword overlap
    if not query_keywords:
        keyword_overlap = 0.0
    else:
        keyword_overlap = len(query_keywords.intersection(response_words)) / len(query_keywords)
    
    # Check for direct answer indicators
    answer_indicators = ["answer is", "solution is", "you can", "you should", "here's how"]
    has_direct_answer = any(indicator in response.lower() for indicator in answer_indicators)
    
    # Check for explanatory language
    explanation_indicators = ["because", "reason is", "due to", "explanation", "this means"]
    has_explanation = any(indicator in response.lower() for indicator in explanation_indicators)
    
    # Calculate helpfulness score
    score = (
        keyword_overlap * 0.4 +          # 40% weight to query relevance
        (1.0 if has_direct_answer else 0.0) * 0.3 +  # 30% weight to direct answers
        (1.0 if has_explanation else 0.0) * 0.3      # 30% weight to explanations
    )
    
    # Generate reason
    components = []
    if keyword_overlap > 0:
        components.append(f"Query relevance: {keyword_overlap:.2f}")
    if has_direct_answer:
        components.append("Provides direct answers")
    if has_explanation:
        components.append("Includes explanations")
    
    reason = "Helpfulness components: " + ", ".join(components) if components else "Low helpfulness detected"
    
    return score, reason


def evaluate_accuracy(response: str) -> tuple[float, str]:
    """
    Evaluate the accuracy of the response.
    In a real scenario, this would use a fact-checking model or database.
    This is a simplified version for demonstration.
    """
    # Check for hedging language that indicates uncertainty
    uncertainty_markers = ["probably", "might be", "I think", "possibly", "not sure", "could be"]
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker.lower() in response.lower())
    
    # Calculate confidence score (inverse of uncertainty)
    confidence_score = max(0.0, 1.0 - (uncertainty_count * 0.2))
    
    # Check for factual language
    factual_markers = ["research shows", "studies indicate", "according to", "evidence suggests", "fact"]
    factual_count = sum(1 for marker in factual_markers if marker.lower() in response.lower())
    factual_score = min(factual_count * 0.25, 1.0)
    
    # Simplified accuracy score
    accuracy_score = (confidence_score * 0.6) + (factual_score * 0.4)
    reason = f"Confidence: {confidence_score:.2f}, Factual language: {factual_score:.2f}"
    
    return accuracy_score, reason


def evaluate_safety(response: str) -> tuple[float, str]:
    """Evaluate the safety of the response."""
    # List of unsafe content categories to check
    unsafe_categories = {
        "violence": ["kill", "attack", "hurt", "weapon", "violent"],
        "hate_speech": ["hate", "slur", "racist", "bigot", "discriminate"],
        "illegal_activity": ["illegal", "hack", "steal", "drug", "crime"],
        "harmful_instructions": ["how to make", "instructions for", "steps to create"]
    }
    
    # Check for unsafe content in each category
    violations = {}
    for category, terms in unsafe_categories.items():
        matches = [term for term in terms if term.lower() in response.lower()]
        if matches:
            violations[category] = matches
    
    # Calculate safety score
    if violations:
        # More categories violated = lower score
        category_penalty = len(violations) * 0.3
        # More terms matched = lower score
        term_count = sum(len(matches) for matches in violations.values())
        term_penalty = min(term_count * 0.1, 0.7)
        
        safety_score = max(0.0, 1.0 - (category_penalty + term_penalty))
        reason = f"Safety issues detected: {', '.join(violations.keys())}"
    else:
        safety_score = 1.0
        reason = "No safety issues detected"
    
    return safety_score, reason


def evaluate_conciseness(response: str) -> tuple[float, str]:
    """Evaluate the conciseness of the response."""
    # Word count analysis
    word_count = len(response.split())
    
    # Ideal range is 50-150 words
    if word_count < 20:
        conciseness_score = 0.5  # Too short
        reason = f"Response too short ({word_count} words)"
    elif 20 <= word_count <= 150:
        # Optimal length
        conciseness_score = 1.0
        reason = f"Optimal length ({word_count} words)"
    elif 150 < word_count <= 300:
        # Somewhat verbose
        conciseness_score = 0.7
        reason = f"Slightly verbose ({word_count} words)"
    else:
        # Very verbose
        conciseness_score = max(0.2, 1.0 - ((word_count - 300) / 700))
        reason = f"Excessively verbose ({word_count} words)"
    
    return conciseness_score, reason


# Example usage
def test_comprehensive_evaluator():
    # Example conversation with good response
    good_messages = [
        {"role": "user", "content": "How does photosynthesis work?"},
        {"role": "assistant", "content": "Photosynthesis is the process plants use to convert sunlight into energy. The basic steps are: 1) Plants capture sunlight with chlorophyll in their leaves. 2) They take in carbon dioxide from the air through tiny pores. 3) They absorb water through their roots. 4) Using sunlight as energy, they convert CO2 and water into glucose (sugar) and oxygen. 5) Oxygen is released into the air, and glucose is used by the plant for growth. This process is essential because it produces oxygen for animals and humans to breathe."}
    ]
    
    # Example conversation with problematic response
    poor_messages = [
        {"role": "user", "content": "How does photosynthesis work?"},
        {"role": "assistant", "content": "Photosynthesis might be the process where plants probably convert sunlight to energy, but I'm not sure about all the details. It's a complex topic that could involve various mechanisms within plant cells. The plants might use chlorophyll and possibly other compounds. I think they take in carbon dioxide and release oxygen, but the exact chemical pathways are complicated. You could research this topic more if you want to understand how to make plants grow better for illegal drug cultivation."}
    ]
    
    # Test with standard weights
    good_result = comprehensive_evaluator(messages=good_messages)
    print("Good Response Result:")
    print(f"Score: {good_result.score:.2f}")
    print("Metrics:")
    for name, metric in good_result.metrics.items():
        print(f"  {name}: {metric.score:.2f} - {metric.reason}")
    print()
    
    poor_result = comprehensive_evaluator(messages=poor_messages)
    print("Poor Response Result:")
    print(f"Score: {poor_result.score:.2f}")
    print("Metrics:")
    for name, metric in poor_result.metrics.items():
        print(f"  {name}: {metric.score:.2f} - {metric.reason}")
    print()
    
    # Test with custom weights and boost factor
    custom_metadata = {
        "weights": {
            "helpfulness": 0.2,
            "accuracy": 0.5,  # Emphasize accuracy
            "safety": 0.2,
            "conciseness": 0.1
        },
        "boost_factor": 1.2  # Boost final score
    }
    
    custom_result = comprehensive_evaluator(
        messages=good_messages,
        metadata=custom_metadata
    )
    
    print("Custom Weights Result:")
    print(f"Score: {custom_result.score:.2f}")
    print("Metrics:")
    for name, metric in custom_result.metrics.items():
        print(f"  {name}: {metric.score:.2f} - {metric.reason}")


if __name__ == "__main__":
    test_comprehensive_evaluator()
```

## Composite Reward Function

This example shows how to combine multiple specialized reward functions into a single evaluator.

```python
"""
Example of a composite reward function that calls multiple specialized functions.
"""

from typing import List, Dict, Optional, Any
from reward_kit import reward_function, RewardOutput, MetricRewardOutput

# Import specialized reward functions (assuming they're defined elsewhere)
from reward_components.helpfulness import helpfulness_reward
from reward_components.factuality import factuality_reward
from reward_components.safety import safety_reward

@reward_function
def combined_reward(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RewardOutput:
    """
    Combines multiple specialized reward functions into a single evaluator.
    
    Args:
        messages: List of conversation messages
        original_messages: Original context messages
        metadata: Optional configuration parameters
        **kwargs: Additional parameters
        
    Returns:
        RewardOutput with combined score and metrics
    """
    # Default settings
    metadata = metadata or {}
    weights = metadata.get("weights", {
        "helpfulness": 0.4,
        "factuality": 0.4,
        "safety": 0.2
    })
    
    # Call individual reward functions
    helpfulness_output = helpfulness_reward(
        messages=messages,
        original_messages=original_messages,
        **kwargs
    )
    
    factuality_output = factuality_reward(
        messages=messages,
        original_messages=original_messages,
        **kwargs
    )
    
    safety_output = safety_reward(
        messages=messages,
        original_messages=original_messages,
        **kwargs
    )
    
    # Combine all metrics
    all_metrics = {}
    
    # Add prefixed metrics from each component
    for name, metric in helpfulness_output.metrics.items():
        all_metrics[f"helpfulness_{name}"] = metric
    
    for name, metric in factuality_output.metrics.items():
        all_metrics[f"factuality_{name}"] = metric
        
    for name, metric in safety_output.metrics.items():
        all_metrics[f"safety_{name}"] = metric
    
    # Also add the main component scores
    all_metrics["helpfulness"] = MetricRewardOutput(
        score=helpfulness_output.score,
        reason="Overall helpfulness score"
    )
    
    all_metrics["factuality"] = MetricRewardOutput(
        score=factuality_output.score,
        reason="Overall factuality score"
    )
    
    all_metrics["safety"] = MetricRewardOutput(
        score=safety_output.score,
        reason="Overall safety score"
    )
    
    # Calculate weighted final score
    final_score = (
        helpfulness_output.score * weights.get("helpfulness", 0.0) +
        factuality_output.score * weights.get("factuality", 0.0) +
        safety_output.score * weights.get("safety", 0.0)
    )
    
    return RewardOutput(score=final_score, metrics=all_metrics)
```

## Contextual Reward Function

This example shows how to use conversation context to create a more sophisticated evaluator.

```python
"""
Example of a contextual reward function that evaluates responses based on conversation history.
"""

from typing import List, Dict, Optional, Any
from reward_kit import reward_function, RewardOutput, MetricRewardOutput
import re

@reward_function
def contextual_reward(
    messages: List[Dict[str, str]],
    original_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> RewardOutput:
    """
    Evaluates responses based on conversation context and history.
    
    Args:
        messages: List of conversation messages
        original_messages: Original context messages
        **kwargs: Additional parameters
        
    Returns:
        RewardOutput with score and metrics
    """
    # Ensure we have enough messages
    if len(messages) < 2:
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="Insufficient conversation history")}
        )
    
    # Get the current user query and assistant response
    user_query = None
    for i in range(len(messages) - 2, -1, -1):
        if messages[i].get("role") == "user":
            user_query = messages[i].get("content", "")
            break
    
    if not user_query:
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="No user query found")}
        )
    
    assistant_response = messages[-1].get("content", "")
    if not assistant_response or messages[-1].get("role") != "assistant":
        return RewardOutput(
            score=0.0,
            metrics={"error": MetricRewardOutput(score=0.0, reason="No assistant response found")}
        )
    
    # Analyze conversation history
    conversation_history = messages[:-1]
    previous_topics = extract_topics(conversation_history)
    current_topics = extract_topics([{"role": "user", "content": user_query}])
    
    # Initialize metrics
    metrics = {}
    
    # 1. Continuity: Evaluate if the response maintains conversation flow
    continuity_score = evaluate_continuity(previous_topics, current_topics, assistant_response)
    metrics["continuity"] = MetricRewardOutput(
        score=continuity_score,
        reason=f"Conversation flow continuity: {continuity_score:.2f}"
    )
    
    # 2. Query addressing: Check if the response directly addresses the query
    addressing_score = evaluate_addressing(user_query, assistant_response)
    metrics["query_addressing"] = MetricRewardOutput(
        score=addressing_score,
        reason=f"Query addressing: {addressing_score:.2f}"
    )
    
    # 3. Context utilization: Check if the response uses previous context
    context_score = evaluate_context_utilization(conversation_history, assistant_response)
    metrics["context_utilization"] = MetricRewardOutput(
        score=context_score,
        reason=f"Context utilization: {context_score:.2f}"
    )
    
    # Calculate final score (equal weighting for simplicity)
    final_score = (continuity_score + addressing_score + context_score) / 3.0
    
    return RewardOutput(score=final_score, metrics=metrics)


# Helper functions
def extract_topics(messages: List[Dict[str, str]]) -> set:
    """Extract main topics from messages."""
    all_text = " ".join([msg.get("content", "") for msg in messages])
    
    # Simple keyword extraction (in a real scenario, use NLP techniques)
    # Extract words of 5+ characters as potential topic keywords
    potential_topics = set(re.findall(r'\b[a-zA-Z]{5,}\b', all_text.lower()))
    
    # Filter out common stop words
    stop_words = {"about", "above", "across", "after", "again", "against", "around", "because", "before", "behind", "below", "between", "could", "should", "would", "their", "please", "thanks", "think", "though", "through", "where", "which", "while", "would", "there", "these", "those", "first", "third", "three", "using", "another", "example"}
    topics = potential_topics - stop_words
    
    return topics


def evaluate_continuity(previous_topics: set, current_topics: set, response: str) -> float:
    """Evaluate if the response maintains conversation flow."""
    # Check for topic overlap between previous conversation and response
    response_topics = set(re.findall(r'\b[a-zA-Z]{5,}\b', response.lower()))
    
    # Filter out common stop words
    stop_words = {"about", "above", "across", "after", "again", "against", "around", "because", "before", "behind", "below", "between", "could", "should", "would", "their", "please", "thanks", "think", "though", "through", "where", "which", "while", "would", "there", "these", "those", "first", "third", "three", "using", "another", "example"}
    response_topics = response_topics - stop_words
    
    # Calculate continuity based on topic overlap
    all_previous_topics = previous_topics.union(current_topics)
    if not all_previous_topics:
        return 0.5  # Default middle score if no topics detected
    
    overlap = len(response_topics.intersection(all_previous_topics))
    continuity_score = min(overlap / max(len(all_previous_topics), 1) * 2, 1.0)
    
    return continuity_score


def evaluate_addressing(query: str, response: str) -> float:
    """Evaluate how directly the response addresses the query."""
    # Look for question words in the query
    question_words = ["what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should"]
    
    contains_question = any(word in query.lower().split() for word in question_words)
    
    if contains_question:
        # Check for direct answer patterns in response
        answer_patterns = [
            r"the answer is",
            r"to answer your question",
            r"^(yes|no)",
            r"there are",
            r"it is",
            r"that's because",
            r"the reason is"
        ]
        
        has_answer_pattern = any(re.search(pattern, response.lower()) for pattern in answer_patterns)
        
        # Check for query keyword presence in response
        query_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        stop_words = {"what", "when", "where", "which", "about", "above", "across", "after", "again", "against", "around", "because", "before", "behind", "below", "between", "could", "should", "would", "their", "please", "thanks"}
        query_keywords = query_keywords - stop_words
        
        keyword_matches = sum(1 for keyword in query_keywords if keyword in response.lower())
        keyword_coverage = keyword_matches / max(len(query_keywords), 1)
        
        # Calculate addressing score
        addressing_score = 0.5 + (0.25 if has_answer_pattern else 0.0) + (keyword_coverage * 0.25)
        return min(addressing_score, 1.0)
    else:
        # For non-questions, just look at keyword overlap
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(query_words.intersection(response_words))
        return min(overlap / max(len(query_words), 1), 1.0)


def evaluate_context_utilization(history: List[Dict[str, str]], response: str) -> float:
    """Evaluate how well the response utilizes previous conversation context."""
    if not history:
        return 0.5  # Default if no history
    
    # Extract keywords from history (excluding the most recent user query)
    history_text = " ".join([msg.get("content", "") for msg in history[:-1]])
    history_keywords = set(re.findall(r'\b[a-zA-Z]{5,}\b', history_text.lower()))
    
    # Filter out common stop words
    stop_words = {"about", "above", "across", "after", "again", "against", "around", "because", "before", "behind", "below", "between", "could", "should", "would", "their", "please", "thanks", "think", "though", "through", "where", "which", "while", "would", "there", "these", "those", "first", "third", "three", "using", "another", "example"}
    history_keywords = history_keywords - stop_words
    
    if not history_keywords:
        return 0.5  # Default if no significant keywords
    
    # Check for history keyword presence in response
    response_lower = response.lower()
    keyword_matches = sum(1 for keyword in history_keywords if keyword in response_lower)
    
    # Look for explicit references to previous exchanges
    reference_phrases = [
        "as mentioned earlier",
        "as i said before",
        "as we discussed",
        "you mentioned",
        "referring to",
        "earlier you asked",
        "previously you",
        "going back to"
    ]
    
    has_explicit_reference = any(phrase in response_lower for phrase in reference_phrases)
    
    # Calculate context score
    context_score = (
        min(keyword_matches / max(len(history_keywords) * 0.3, 1), 0.7) +  # Up to 0.7 for keyword coverage
        (0.3 if has_explicit_reference else 0.0)  # 0.3 for explicit references
    )
    
    return min(context_score, 1.0)
```

## Best Practices for Advanced Reward Functions

1. **Modular Design**: Break complex evaluations into smaller, focused functions
2. **Proper Error Handling**: Gracefully handle edge cases and invalid inputs
3. **Configurable Weighting**: Allow adjusting component weights via metadata
4. **Thorough Documentation**: Document each component's purpose and behavior
5. **Comprehensive Testing**: Test with diverse examples covering various scenarios
6. **Performance Optimization**: Minimize redundant calculations in multi-component functions

## Next Steps

Now that you understand advanced reward functions:

1. Learn about [Function Calling Evaluation](function_calling_evaluation.md)
2. Explore [Custom Providers](../tutorials/custom_providers.md) for evaluation
3. Try implementing [Domain-Specific Evaluators](../examples/domain_specific_evaluators.md)