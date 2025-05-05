# Reward Functions Overview

This guide provides an overview of all out-of-the-box reward functions available in the Reward Kit library.

## Introduction

Reward Kit includes several pre-built reward functions for common evaluation tasks. These functions can be used directly or as building blocks for more complex evaluations.

## Available Reward Functions

### Format and Structure Rewards

These reward functions evaluate the format and structure of responses.

- **Format Reward**: Evaluate responses against a regex pattern (e.g., `<think>...</think><answer>...</answer>`)
  ```python
  from reward_kit.rewards.format import format_reward
  
  result = format_reward(
      messages=messages,
      pattern=r"^<think>\n.*?</think>\n<answer>\n.*?</answer>$",
      flags=re.DOTALL
  )
  ```

- **Tag Count Reward**: Check for exactly one of each specified tag
  ```python
  from reward_kit.rewards.tag_count import tag_count_reward
  
  result = tag_count_reward(
      messages=messages,
      tags=["pros", "cons"]
  )
  ```

### Accuracy and Correctness Rewards

These reward functions evaluate the accuracy of responses against expected answers.

- **Accuracy Reward**: Compare answers to ground truth
  ```python
  from reward_kit.rewards.accuracy import accuracy_reward
  
  result = accuracy_reward(
      messages=messages,
      ground_truth="Paris"
  )
  ```

- **Math Reward**: Compare numerical answers with expected values
  ```python
  from reward_kit.rewards.math import math_reward
  
  result = math_reward(
      messages=messages,
      expected_answer="42"
  )
  ```

### Language and Style Rewards

These reward functions evaluate linguistic aspects of responses.

- **Language Consistency Reward**: Ensure response is in the target language
  ```python
  from reward_kit.rewards.language_consistency import language_consistency_reward
  
  result = language_consistency_reward(
      messages=messages,
      target_language="spanish"
  )
  ```

- **Reasoning Steps Reward**: Encourage step-by-step reasoning
  ```python
  from reward_kit.rewards.reasoning_steps import reasoning_steps_reward
  
  result = reasoning_steps_reward(
      messages=messages,
      min_steps=3
  )
  ```

### Length and Verbosity Rewards

These reward functions evaluate the length and verbosity of responses.

- **Length Reward**: Evaluate response against length targets
  ```python
  from reward_kit.rewards.length import length_reward
  
  result = length_reward(
      messages=messages,
      target_length=200,  # Target token count
      token_method="whitespace"
  )
  ```

- **Cosine Length Reward**: Scale rewards based on length using cosine schedule
  ```python
  from reward_kit.rewards.length import cosine_length_reward
  
  result = cosine_length_reward(
      messages=messages,
      correctness=0.9,  # High correctness score
      max_length=500,
      min_value_correct=0.5,
      max_value_correct=1.0
  )
  ```

- **Repetition Penalty Reward**: Penalize repetitive content
  ```python
  from reward_kit.rewards.repetition import repetition_penalty_reward
  
  result = repetition_penalty_reward(
      messages=messages,
      max_penalty=0.5,
      ngram_size=3
  )
  ```

### Code Execution Rewards

These reward functions evaluate code by running it and comparing the output to expected results.

- **Binary Code Reward**: Binary pass/fail for code execution
  ```python
  from reward_kit.rewards.code_execution import binary_code_reward
  
  result = binary_code_reward(
      messages=messages,
      expected_output="expected result",
      language="python"
  )
  ```

- **Fractional Code Reward**: Return exact pass rate for code execution
  ```python
  from reward_kit.rewards.code_execution import fractional_code_reward
  
  result = fractional_code_reward(
      messages=messages,
      test_cases=[
          {"input": "arg1", "expected_output": "result1"},
          {"input": "arg2", "expected_output": "result2"}
      ],
      language="python"
  )
  ```

- **IOI C/C++ Code Reward**: Evaluate C/C++ code using Piston engine
  ```python
  from reward_kit.rewards.cpp_code import ioi_cpp_code_reward
  
  result = ioi_cpp_code_reward(
      messages=messages,
      test_cases=[
          {"input": "4\n5", "expected_output": "9"},
          {"input": "10\n20", "expected_output": "30"}
      ],
      language="cpp"  # or "c"
  )
  ```

- **Binary C/C++ Code Reward**: Binary pass/fail for C/C++ code
  ```python
  from reward_kit.rewards.cpp_code import binary_cpp_code_reward
  
  result = binary_cpp_code_reward(
      messages=messages,
      test_cases=[
          {"input": "4\n5", "expected_output": "9"}
      ],
      language="cpp"
  )
  ```

### Function Calling Rewards

These reward functions evaluate function calls in LLM responses against expected schemas and behaviors.

- **Schema Jaccard Reward**: Compare function calls to expected schema
  ```python
  from reward_kit.rewards.function_calling import schema_jaccard_reward
  
  result = schema_jaccard_reward(
      messages=messages,
      expected_schema=schema
  )
  ```

- **LLM Judge Reward**: Use an LLM to evaluate function call quality
  ```python
  from reward_kit.rewards.function_calling import llm_judge_reward
  
  result = llm_judge_reward(
      messages=messages,
      expected_schema=schema,
      expected_behavior=behavior_description
  )
  ```

- **Composite Function Call Reward**: Combine schema validation and LLM judgment
  ```python
  from reward_kit.rewards.function_calling import composite_function_call_reward
  
  result = composite_function_call_reward(
      messages=messages,
      expected_schema=schema,
      expected_behavior=behavior_description
  )
  ```

### JSON Schema Rewards

These reward functions validate JSON outputs against predefined schemas.

- **JSON Schema Reward**: Validate JSON against a schema
  ```python
  from reward_kit.rewards.json_schema import json_schema_reward
  
  result = json_schema_reward(
      messages=messages,
      schema=json_schema
  )
  ```

### Combined Metrics Rewards

These reward functions combine multiple evaluation aspects into a single score.

- **Cosine-Scaled Accuracy + Length Reward**: Combine accuracy with length efficiency
  ```python
  from reward_kit.rewards.accuracy_length import cosine_scaled_accuracy_length_reward
  
  result = cosine_scaled_accuracy_length_reward(
      messages=messages,
      ground_truth="Paris",
      max_length=200,
      correctness_weight=0.7,
      length_weight=0.3
  )
  ```

## Choosing the Right Reward Function

Here's a guide to help you choose the appropriate reward function for your task:

| Task | Recommended Reward Function |
|------|----------------------------|
| Evaluating format adherence | `format_reward` |
| Checking tag usage and structure | `tag_count_reward` |
| Evaluating factual accuracy | `accuracy_reward` |
| Ensuring consistent language | `language_consistency_reward` |
| Encouraging step-by-step reasoning | `reasoning_steps_reward` |
| Controlling response length | `length_reward` |
| Optimizing for brevity and correctness | `cosine_scaled_accuracy_length_reward` |
| Reducing repetition | `repetition_penalty_reward` |
| Evaluating Python code | `fractional_code_reward` or `binary_code_reward` |
| Evaluating C/C++ code | `ioi_cpp_code_reward` or `binary_cpp_code_reward` |
| Validating tool use and function calls | `composite_function_call_reward` |
| Checking structured data outputs | `json_schema_reward` |
| Evaluating mathematical solutions | `math_reward` |

## Combining Reward Functions

You can combine multiple reward functions to create comprehensive evaluations:

```python
from reward_kit.rewards.accuracy import accuracy_reward
from reward_kit.rewards.length import length_reward
from reward_kit import reward_function, RewardOutput, MetricRewardOutput

@reward_function
def combined_accuracy_length(messages, ground_truth=None, **kwargs):
    """Combine accuracy and length evaluation."""
    # Check accuracy
    accuracy_result = accuracy_reward(
        messages=messages,
        ground_truth=ground_truth
    )
    
    # Check length
    length_result = length_reward(
        messages=messages,
        target_length=150
    )
    
    # Combine scores with weighting
    # 70% accuracy, 30% length
    combined_score = 0.7 * accuracy_result["score"] + 0.3 * length_result["score"]
    
    # Combine metrics
    metrics = {
        "accuracy": MetricRewardOutput(
            score=accuracy_result["score"],
            reason=accuracy_result["reason"]
        ),
        "length": MetricRewardOutput(
            score=length_result["score"],
            reason=length_result["reason"]
        )
    }
    
    return RewardOutput(score=combined_score, metrics=metrics)
```

## Pre-Built Combined Metrics

Reward Kit offers pre-built functions that combine multiple metrics:

- **Cosine-Scaled Accuracy + Length**: Combines accuracy with length using a cosine schedule
  ```python
  from reward_kit.rewards.accuracy_length import cosine_scaled_accuracy_length_reward
  
  result = cosine_scaled_accuracy_length_reward(
      messages=messages,
      ground_truth="Paris",
      max_length=200,
      correctness_weight=0.7,  # Weight for accuracy component
      length_weight=0.3        # Weight for length component
  )
  ```
  
  This function:
  - Evaluates response accuracy against ground truth
  - Measures response length efficiency using a cosine schedule
  - Rewards shorter correct answers more than longer ones
  - Maintains a clear separation between correct and incorrect answers
  - Allows customizable weighting between accuracy and length

## Next Steps

- Explore individual reward function documentation:
  - [Format and Structure Rewards](../api_reference/reward_functions/format.md)
  - [Accuracy and Correctness Rewards](../api_reference/reward_functions/accuracy.md)
  - [Language and Style Rewards](../api_reference/reward_functions/language.md)
  - [Length and Verbosity Rewards](../api_reference/reward_functions/length.md)
  - [Code Execution Rewards](code_execution_evaluation.md)
  - [Function Calling Rewards](function_calling_evaluation.md)
  - [JSON Schema Validation](json_schema_validation.md)
  - [Math Evaluation](math_evaluation.md)
  - [Combined Metrics Rewards](../api_reference/reward_functions/combined.md)
- Learn how to [create your own reward functions](../tutorials/creating_your_first_reward_function.md)
- Read [best practices](../tutorials/best_practices.md) for effective evaluations
- See [examples](../developer_guide/evaluation_workflows.md) of common evaluation workflows