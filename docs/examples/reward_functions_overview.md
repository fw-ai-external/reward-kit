# Reward Functions Overview

This guide provides an overview of all out-of-the-box reward functions available in the Reward Kit library.

## Introduction

Reward Kit includes several pre-built reward functions for common evaluation tasks. These functions can be used directly or as building blocks for more complex evaluations.

## Available Reward Functions

### Code Execution Rewards

The code execution reward functions evaluate code by running it and comparing the output to expected results.

- **[Local Code Execution](code_execution_evaluation.md)**: Execute code securely on your local machine
  ```python
  from reward_kit.rewards.code_execution import local_code_execution_reward
  
  result = local_code_execution_reward(
      messages=messages,
      expected_output="expected result",
      language="python"
  )
  ```

- **[E2B Code Execution](code_execution_with_e2b.md)**: Execute code in a secure cloud sandbox
  ```python
  from reward_kit.rewards.code_execution import e2b_code_execution_reward
  
  result = e2b_code_execution_reward(
      messages=messages,
      expected_output="expected result",
      language="python",
      api_key="your_e2b_api_key"
  )
  ```

### Function Calling Rewards

These reward functions evaluate function calls in LLM responses against expected schemas and behaviors.

- **[Schema Jaccard Reward](function_calling_evaluation.md#schema-jaccard-reward)**: Compare function calls to expected schema
  ```python
  from reward_kit.rewards.function_calling import schema_jaccard_reward
  
  result = schema_jaccard_reward(
      messages=messages,
      expected_schema=schema
  )
  ```

- **[LLM Judge Reward](function_calling_evaluation.md#llm-judge-reward)**: Use an LLM to evaluate function call quality
  ```python
  from reward_kit.rewards.function_calling import llm_judge_reward
  
  result = llm_judge_reward(
      messages=messages,
      expected_schema=schema,
      expected_behavior=behavior_description
  )
  ```

- **[Composite Function Call Reward](function_calling_evaluation.md#composite-function-call-reward)**: Combine schema validation and LLM judgment
  ```python
  from reward_kit.rewards.function_calling import composite_function_call_reward
  
  result = composite_function_call_reward(
      messages=messages,
      expected_schema=schema,
      expected_behavior=behavior_description
  )
  ```

### JSON Schema Rewards

The JSON schema reward functions validate JSON outputs against predefined schemas.

- **[JSON Schema Reward](json_schema_validation.md)**: Validate JSON against a schema
  ```python
  from reward_kit.rewards.json_schema import json_schema_reward
  
  result = json_schema_reward(
      messages=messages,
      schema=json_schema
  )
  ```

- **[Validate JSON String](json_schema_validation.md#direct-json-string-validation)**: Direct validation of JSON strings
  ```python
  from reward_kit.rewards.json_schema import validate_json_string
  
  result = validate_json_string(
      json_str=json_string,
      schema=json_schema
  )
  ```

### Math Rewards

The math reward functions evaluate mathematical answers in LLM responses.

- **[Math Reward](math_evaluation.md)**: Compare numerical answers with expected values
  ```python
  from reward_kit.rewards.math import math_reward
  
  result = math_reward(
      messages=messages,
      expected_answer="42"
  )
  ```

## Choosing the Right Reward Function

Here's a guide to help you choose the appropriate reward function for your task:

| Task | Recommended Reward Function |
|------|----------------------------|
| Evaluating coding solutions | `local_code_execution_reward` or `e2b_code_execution_reward` |
| Validating tool use and function calls | `composite_function_call_reward` |
| Checking structured data outputs | `json_schema_reward` |
| Evaluating mathematical solutions | `math_reward` |
| Simple schema validation of function calls | `schema_jaccard_reward` |
| Qualitative evaluation of function calls | `llm_judge_reward` |

## Combining Reward Functions

You can combine multiple reward functions to create comprehensive evaluations:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward
from reward_kit.rewards.json_schema import json_schema_reward
from reward_kit import reward_function

@reward_function
def combined_code_json_reward(messages, original_messages=None, **kwargs):
    # First, check if code executes correctly
    code_result = local_code_execution_reward(
        messages=messages,
        expected_output="[1, 2, 3]",
        language="python"
    )
    
    # Then, check if the output is valid JSON
    json_schema = {
        "type": "array",
        "items": {"type": "integer"}
    }
    
    json_result = json_schema_reward(
        messages=messages,
        schema=json_schema
    )
    
    # Combine the metrics
    metrics = {
        "code_execution": code_result.metrics.get("execution_result"),
        "json_validity": json_result.metrics.get("validation")
    }
    
    # Calculate overall score (50% code execution, 50% JSON validity)
    overall_score = 0.5 * code_result.score + 0.5 * json_result.score
    
    return {"score": overall_score, "metrics": metrics}
```

## Next Steps

- Explore individual reward function documentation:
  - [Code Execution Evaluation](code_execution_evaluation.md)
  - [Code Execution with E2B](code_execution_with_e2b.md)
  - [Function Calling Evaluation](function_calling_evaluation.md)
  - [JSON Schema Validation](json_schema_validation.md)
  - [Math Evaluation](math_evaluation.md)
- Learn how to [create your own reward functions](../tutorials/creating_your_first_reward_function.md)
- Read [best practices](../tutorials/best_practices.md) for effective evaluations
- See [examples](../developer_guide/evaluation_workflows.md) of common evaluation workflows