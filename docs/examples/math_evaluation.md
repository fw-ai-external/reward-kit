# Math Evaluation

This guide demonstrates how to evaluate mathematical answers in LLM responses using the math reward functions.

## Overview

The `math_reward` function allows you to:

1. Extract numerical answers from LLM responses
2. Compare them with expected answers or reference solutions
3. Handle various formats including fractions, decimals, and scientific notation
4. Support LaTeX formatted answers in markdown

## Prerequisites

Before using the math evaluation rewards, ensure you have:

1. **Python 3.8+** installed on your system
2. **Reward Kit** installed: `pip install reward-kit`

## Basic Usage

Here's a simple example of how to use the math reward function:

```python
from reward_kit.rewards.math import math_reward

# Example conversation with a math problem
messages = [
    {
        "role": "user", 
        "content": "Calculate 15% of 80."
    },
    {
        "role": "assistant", 
        "content": "To calculate 15% of 80, I'll multiply 80 by 0.15:\n\n80 × 0.15 = 12\n\nTherefore, 15% of 80 is 12."
    }
]

# Expected answer
expected_answer = "12"

# Evaluate the response
result = math_reward(
    messages=messages,
    expected_answer=expected_answer
)

# Print the results
print(f"Score: {result.score}")
print("Metrics:")
for name, metric in result.metrics.items():
    print(f"  {name}: {metric.score}")
    print(f"    {metric.reason}")
```

## How It Works

The math reward function:

1. Extracts potential answer values from the last assistant message
2. Extracts expected answer value from the provided string
3. Compares them with tolerance for floating-point values
4. Returns a score of 1.0 for correct answers and 0.0 for incorrect answers
5. Provides detailed metrics about the extraction and comparison process

## Supported Answer Formats

The math reward function can extract and compare answers in various formats:

### Integer and Decimal Numbers

```
42
-27
3.14159
0.5
```

### Fractions

```
3/4
-5/8
1 2/3 (mixed fractions)
```

### Scientific Notation

```
1.23e4
6.022 × 10^23
5.67 × 10⁻⁸
```

### LaTeX Formatting

```
\boxed{42}
\frac{3}{4}
\frac{22}{7} \approx 3.14
\pi \approx 3.14159
2.998 \times 10^8 \text{ m/s}
```

### Units

```
42 kg
3.14 m/s²
5 \text{ meters}
```

## Advanced Usage

### Customizing Extraction

You can customize the extraction process to look for answers in particular formats or locations:

```python
from reward_kit.rewards.math import math_reward

# Messages with LaTeX formatted answer
messages = [
    {
        "role": "user", 
        "content": "What is the area of a circle with radius 3 cm?"
    },
    {
        "role": "assistant", 
        "content": "To find the area of a circle, I'll use the formula:\n\nArea = πr²\n\nSubstituting r = 3 cm:\n\nArea = π × 3² = 9π cm²\n\nCalculating with π ≈ 3.14159:\n\nArea ≈ 28.27 cm²\n\nTherefore, the area of a circle with radius 3 cm is \n\n$$\\boxed{28.27 \\text{ cm}^2}$$"
    }
]

# Evaluate with custom extraction patterns
result = math_reward(
    messages=messages,
    expected_answer="28.27 cm^2",
    extract_boxed_only=True,  # Only look for answers in \boxed{} environments
    ignore_units=False,       # Consider units in the comparison
    tolerance=0.01            # Allow for slight differences in rounding
)
```

### Multiple Valid Answers

Sometimes, multiple forms of the same answer are acceptable. You can evaluate against multiple correct answers:

```python
from reward_kit.rewards.math import math_reward

# Message with fraction answer
messages = [
    {
        "role": "user", 
        "content": "What is 1/4 + 1/6?"
    },
    {
        "role": "assistant", 
        "content": "To add fractions with different denominators, I need to find a common denominator.\n\n1/4 + 1/6\n\nLCD = 12\n\n1/4 = 3/12\n1/6 = 2/12\n\n3/12 + 2/12 = 5/12\n\nTherefore, 1/4 + 1/6 = 5/12"
    }
]

# Accept either fraction or decimal form
result = math_reward(
    messages=messages,
    expected_answer=["5/12", "0.41666"], # Accept either form
    tolerance=0.001  # Small tolerance for decimal approximation
)
```

### Original Messages as Reference

If the correct answer is in the original messages, you can extract it automatically:

```python
from reward_kit.rewards.math import math_reward

# Original conversation with correct answer
original_messages = [
    {
        "role": "user", 
        "content": "Solve the equation 2x + 5 = 15. The answer is x = 5."
    }
]

# Generated response to evaluate
generated_messages = [
    {
        "role": "user", 
        "content": "Solve the equation 2x + 5 = 15."
    },
    {
        "role": "assistant", 
        "content": "To solve the equation 2x + 5 = 15, I'll isolate the variable x.\n\n2x + 5 = 15\n2x = 15 - 5\n2x = 10\nx = 10/2\nx = 5\n\nTherefore, the solution is x = 5."
    }
]

# Extract expected answer from original messages
result = math_reward(
    messages=generated_messages,
    original_messages=original_messages,
    extract_answer_from_original=True  # Extract answer from original messages
)
```

## Use Cases

### Evaluating Math Problem Solving

The math reward function is perfect for evaluating responses to:

- Basic arithmetic problems
- Algebra equations
- Calculus problems
- Physics calculations
- Economics computations
- Statistics problems

### Educational Applications

Use the math reward function to:

- Automatically grade math homework
- Provide instant feedback on practice problems
- Evaluate mathematical reasoning in tutoring systems

## Best Practices

1. **Be Explicit About Units**: Specify whether units should be considered in the comparison
2. **Consider Fractions vs. Decimals**: Decide if approximate decimal answers are acceptable for fraction problems
3. **Set Appropriate Tolerance**: Use a tolerance appropriate for the problem (e.g., higher for complex calculations)
4. **Look for Final Answers**: Set up extraction patterns to focus on the final answer rather than intermediate steps
5. **Multiple Representations**: Consider all valid forms of an answer (fraction, decimal, scientific notation)
6. **LaTeX Handling**: Take advantage of the LaTeX support for nicely formatted answers

## Limitations

- Cannot evaluate the correctness of the solution method, only the final answer
- May have difficulty with extremely complex LaTeX expressions
- Cannot evaluate mathematical proofs or abstract reasoning
- Works best with numerical answers rather than symbolic expressions

## Next Steps

- Learn about [Code Execution Evaluation](code_execution_with_e2b.md) for evaluating code solutions
- Explore [Function Calling Evaluation](function_calling_evaluation.md) for evaluating tool use
- See [Creating Custom Reward Functions](../tutorials/creating_your_first_reward_function.md) to build your own specialized math evaluators