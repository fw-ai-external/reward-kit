"""
Example demonstrating the cosine-scaled accuracy + length reward function.

This example shows how to use the cosine_scaled_accuracy_length_reward function
to evaluate responses based on both accuracy and length efficiency.

Note: The accuracy detection in this example may show unexpected behavior in some cases
due to the text extraction mechanisms. For production use, consider providing custom
extract_fn and compare_fn parameters tailored to your specific content types.
"""

from reward_kit.rewards.accuracy_length import cosine_scaled_accuracy_length_reward


def simple_extract_fn(text: str) -> str:
    """
    Simple function to extract place names for our examples.

    This function is designed specifically for our capital city examples
    and makes it easier to extract the answer.
    """
    # List of cities we're looking for
    cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"]

    # Look for key phrases indicating the actual answer (for incorrect examples)
    if "Lyon is the capital" in text or "believe Lyon is the capital" in text:
        return "Lyon"

    # Check for exact city names in the text
    for city in cities:
        if city in text:
            return city

    # Return empty string if no city found
    return ""


def main():
    """Run the example."""
    # Example 1: Short correct answer
    short_correct_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    # Example 2: Long correct answer
    long_correct_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": (
                "The capital of France is Paris. Paris is located in the north-central "
                "part of the country on the Seine River. It is the largest city in France "
                "and serves as the country's political, cultural, and economic center. "
                "Paris is known for landmarks such as the Eiffel Tower, Louvre Museum, "
                "Notre-Dame Cathedral, and Arc de Triomphe. The city has a population "
                "of about 2.2 million people in the city proper."
            ),
        },
    ]

    # Example 3: Short incorrect answer
    short_incorrect_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Lyon."},
    ]

    # Example 4: Long incorrect answer with showing work
    long_incorrect_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": (
                "I need to identify the capital city of France. France is a country in "
                "Western Europe. The largest cities in France include Paris, Marseille, "
                "Lyon, Toulouse, and Nice. Among these, I believe Lyon is the capital "
                "city of France, as it's centrally located and historically significant."
            ),
        },
    ]

    # Evaluate each example with default parameters
    print("===== Evaluating with Default Parameters =====")
    print("\nShort Correct Answer:")
    evaluate_example(short_correct_messages, "Paris")

    print("\nLong Correct Answer:")
    evaluate_example(long_correct_messages, "Paris")

    print("\nShort Incorrect Answer:")
    evaluate_example(short_incorrect_messages, "Paris")

    print("\nLong Incorrect Answer:")
    evaluate_example(long_incorrect_messages, "Paris")

    # Evaluate with custom parameters
    print("\n\n===== Evaluating with Custom Parameters =====")
    print("\nShort Correct Answer (80% accuracy weight, 20% length weight):")
    evaluate_example(
        short_correct_messages,
        "Paris",
        correctness_weight=0.8,
        length_weight=0.2,
    )


def evaluate_example(messages, ground_truth, **kwargs):
    """Evaluate an example and print the results."""
    # Set reasonable defaults if not specified in kwargs
    params = {
        "max_length": 100,  # Reasonable max_length for examples
        "min_value_wrong": 0.0,
        "max_value_wrong": 0.3,
        "min_value_correct": 0.5,
        "max_value_correct": 1.0,
        "correctness_weight": 0.7,
        "length_weight": 0.3,
        "extract_fn": simple_extract_fn,  # Use our custom extractor
    }
    # Update with any provided kwargs
    params.update(kwargs)

    # Prepare ground_truth in the expected List[Message] format
    # (as List[Dict[str, str]] which the decorator will convert)
    if isinstance(ground_truth, str):
        gt_list = [{"role": "assistant", "content": ground_truth}]
    elif ground_truth is None:
        gt_list = None
    else:
        # Assuming it might already be in the correct list format if not str
        gt_list = ground_truth

    result = cosine_scaled_accuracy_length_reward(
        messages=messages, ground_truth=gt_list, **params
    )

    # Extract the assistant's response for reference
    response = messages[-1]["content"]
    word_count = len(response.split())

    # Print evaluation results
    print(f'Response ({word_count} words): "{response[:50]}..."')
    print(f"Combined Score: {result.score:.2f}")
    print(f"Accuracy Score: {result.metrics['accuracy'].score:.2f}")
    print(f"Length Score: {result.metrics['length'].score:.2f}")


if __name__ == "__main__":
    main()
