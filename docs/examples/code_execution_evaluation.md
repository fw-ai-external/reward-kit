# Code Execution Evaluation

This guide demonstrates how to evaluate code solutions using the Reward Kit's code execution reward functions.

## Overview

The code execution reward functions allow you to:

1. Extract code blocks from LLM responses
2. Execute the code in a secure environment
3. Compare the output with expected results
4. Get detailed execution metrics and error reports

## Prerequisites

Before using the code execution rewards, ensure you have:

1. **Python 3.8+** installed on your system
2. **Reward Kit** installed: `pip install reward-kit`
3. For JavaScript evaluation: **Node.js** installed on your system

## Available Reward Functions

Reward Kit provides two main methods for code execution evaluation:

1. **Local Code Execution**: Executes code securely on your local machine
2. **E2B Code Execution**: Executes code in a cloud sandbox (requires E2B account)

## Local Code Execution

### Basic Usage

Here's a simple example of evaluating Python code:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Example conversation with a coding task
messages = [
    {
        "role": "user",
        "content": "Write a Python function to calculate the factorial of a number."
    },
    {
        "role": "assistant",
        "content": """Here's a Python function to calculate the factorial of a number:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Test the function
print(factorial(5))
```

This function uses recursion to calculate the factorial. For n = 5, it should output 120."""
    }
]

# Evaluate the code
result = local_code_execution_reward(
    messages=messages,
    expected_output="120",
    language="python",
    timeout=5
)

# Print the results
print(f"Score: {result.score}")
print("Metrics:")
for name, metric in result.metrics.items():
    print(f"  {name}: {metric.score}")
    print(f"    {metric.reason}")
```

### How It Works

The local code execution reward function:

1. Extracts code blocks from the last assistant message
2. Creates a secure, isolated environment for execution
3. Runs the code with timeout and resource limits
4. Captures stdout, stderr, and exit status
5. Compares the output with expected results if provided
6. Returns detailed metrics about execution and output matching

### Security Features

The local code execution uses multiple security layers:

- **Process Isolation**: Code runs in a separate process
- **Resource Limits**: Restricts memory usage and CPU time
- **Filesystem Restrictions**: Disables destructive file operations
- **System Call Restrictions**: Prevents access to sensitive system calls
- **Timeout Enforcement**: Terminates long-running code
- **Safe Libraries**: Disables potentially dangerous library functions

### JavaScript Execution

You can also evaluate JavaScript code:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Example with JavaScript code
messages = [
    {
        "role": "user",
        "content": "Write a JavaScript function to check if a string is a palindrome."
    },
    {
        "role": "assistant",
        "content": """Here's a JavaScript function to check if a string is a palindrome:

```javascript
function isPalindrome(str) {
    // Remove non-alphanumeric characters and convert to lowercase
    const cleanStr = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    
    // Compare with its reverse
    const reversedStr = cleanStr.split('').reverse().join('');
    
    return cleanStr === reversedStr;
}

// Test the function
console.log(isPalindrome("A man, a plan, a canal: Panama"));  // Should output true
console.log(isPalindrome("hello"));  // Should output false
```

This function removes any non-alphanumeric characters and converts the string to lowercase before checking if it reads the same forward and backward."""
    }
]

# Evaluate the JavaScript code
result = local_code_execution_reward(
    messages=messages,
    expected_output="true\nfalse",
    language="javascript",
    timeout=5
)
```

### Advanced Options

You can customize the execution with various parameters:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Custom execution parameters
result = local_code_execution_reward(
    messages=messages,
    expected_output="120",
    language="python",
    timeout=10,        # Longer timeout for complex code
    max_memory_mb=200  # Higher memory limit
)
```

### Automatic Expected Output Extraction

If the expected output is mentioned in the conversation, it can be extracted automatically:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Conversation with expected output mentioned in the prompt
messages = [
    {
        "role": "user",
        "content": "Write a function to calculate the sum of numbers from 1 to n. For n=5, the expected output is 15."
    },
    {
        "role": "assistant",
        "content": """Here's a function to calculate the sum of numbers from 1 to n:

```python
def sum_to_n(n):
    return sum(range(1, n+1))

# Test the function
print(sum_to_n(5))
```

This function uses the built-in sum() and range() functions to calculate the sum efficiently."""
    }
]

# Extract expected output from the conversation
result = local_code_execution_reward(
    messages=messages,
    original_messages=messages,  # Provide the original messages for extraction
    language="python"
)
```

## E2B Code Execution

For more information on using E2B for code execution, see the dedicated guide: [Code Execution with E2B](code_execution_with_e2b.md).

## Output Comparison

The code execution reward functions use sophisticated output comparison methods to handle various output formats.

### Exact Matching

For simple outputs, exact matching is used:

```
Expected: "Hello, world!"
Actual: "Hello, world!"
Score: 1.0
```

### Numeric Comparison

For numeric outputs, relative difference is calculated:

```
Expected: "42"
Actual: "42.001"
Score: 0.99  # Very close match
```

### Array/List Comparison

For arrays and lists, both structure and content are compared:

```
Expected: "[1, 2, 3]"
Actual: "[1, 2, 3, 4]"
Score: 0.75  # Partial match
```

### Multiline Text Comparison

For multiline output, line-by-line comparison is used:

```
Expected: "Line 1\nLine 2\nLine 3"
Actual: "Line 1\nLine 2\nLine X"
Score: 0.89  # Most lines match
```

## Use Cases

### Coding Assessment

Evaluate code solutions to programming problems:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Define a coding problem
problem = "Write a function that finds the largest number in a list."
expected_output = "9"

# User's solution
solution = """
def find_largest(numbers):
    return max(numbers)

print(find_largest([5, 2, 9, 3, 7]))
"""

# Create message format
messages = [
    {"role": "user", "content": problem},
    {"role": "assistant", "content": f"```python\n{solution}\n```"}
]

# Evaluate the solution
result = local_code_execution_reward(
    messages=messages,
    expected_output=expected_output,
    language="python"
)

print(f"Score: {result.score}")
```

### Algorithm Comparison

Compare different algorithms for the same problem:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward
import time

# Problem: Find all prime numbers less than 100

# Solution 1: Simple approach
solution1 = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

primes = []
for num in range(2, 100):
    if is_prime(num):
        primes.append(num)
        
print(len(primes))
"""

# Solution 2: Optimized approach
solution2 = """
def sieve_of_eratosthenes(limit):
    primes = []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for num in range(2, limit + 1):
        if sieve[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                sieve[multiple] = False
    
    return primes

print(len(sieve_of_eratosthenes(99)))
"""

# Expected output: 25 prime numbers less than 100
expected_output = "25"

# Evaluate solutions
solutions = [solution1, solution2]
results = []

for i, solution in enumerate(solutions, 1):
    messages = [
        {"role": "user", "content": "Find all prime numbers less than 100 and print the count."},
        {"role": "assistant", "content": f"```python\n{solution}\n```"}
    ]
    
    start_time = time.time()
    result = local_code_execution_reward(
        messages=messages,
        expected_output=expected_output,
        language="python",
        timeout=10
    )
    execution_time = time.time() - start_time
    
    results.append({
        "solution": i,
        "score": result.score,
        "execution_time": execution_time
    })

# Compare results
for res in results:
    print(f"Solution {res['solution']}: Score={res['score']}, Time={res['execution_time']:.4f}s")
```

### Multiple Language Support

Evaluate solutions in different programming languages:

```python
from reward_kit.rewards.code_execution import local_code_execution_reward

# Problem: Check if a number is even
problem = "Write a function to check if a number is even. Test it with the numbers 4 and 7."
expected_output = "true\nfalse"

# Python solution
python_solution = """
def is_even(number):
    return number % 2 == 0

print(is_even(4))
print(is_even(7))
"""

# JavaScript solution
js_solution = """
function isEven(number) {
    return number % 2 === 0;
}

console.log(isEven(4));
console.log(isEven(7));
"""

# Evaluate both solutions
languages = ["python", "javascript"]
solutions = [python_solution, js_solution]

for lang, solution in zip(languages, solutions):
    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": f"```{lang}\n{solution}\n```"}
    ]
    
    result = local_code_execution_reward(
        messages=messages,
        expected_output=expected_output,
        language=lang
    )
    
    print(f"{lang.capitalize()} solution score: {result.score}")
```

## Best Practices

1. **Security First**: Always use the built-in security mechanisms and don't disable them
2. **Timeout Setting**: Choose reasonable timeouts based on task complexity
3. **Expected Output**: Be specific about expected output format for accurate comparison
4. **Error Handling**: Check execution error metrics even when code runs successfully
5. **Resource Limits**: Set appropriate memory limits for the complexity of the code
6. **Test Environment**: Ensure required dependencies are available in the execution environment
7. **Edge Cases**: Test the reward function with a variety of inputs, including edge cases

## Limitations

- Cannot evaluate non-deterministic code reliably
- Limited to languages supported by the reward function (Python and JavaScript for local execution)
- Cannot evaluate code that requires external resources (databases, APIs, etc.) without mocking
- May have limitations with GUI applications or complex I/O operations
- Security mechanisms may prevent some valid code from executing

## Next Steps

- For cloud-based code execution, see [Code Execution with E2B](code_execution_with_e2b.md)
- Learn about [Function Calling Evaluation](function_calling_evaluation.md) for evaluating tool use
- Explore [JSON Schema Validation](json_schema_validation.md) for structured outputs
- See [Creating Custom Reward Functions](../tutorials/creating_your_first_reward_function.md) to build your own evaluators