#!/usr/bin/env python
"""
Test script to verify E2B integration works end-to-end with JavaScript.

This script runs a simple JavaScript code example using the E2B code execution
reward function and verifies that it works correctly.
"""

import os
from reward_kit.rewards.code_execution import e2b_code_execution_reward, _HAS_E2B

def main():
    print("Testing E2B JavaScript integration...")
    
    # Verify E2B package is installed
    if not _HAS_E2B:
        print("Error: E2B package is not installed. Please install it with: pip install e2b")
        return
    
    # Verify API key is available
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        print("Error: E2B_API_KEY environment variable is not set")
        return
    
    print("E2B package installed and API key found.")
    
    # Simple test case with JavaScript code
    messages = [
        {
            "role": "user",
            "content": "Write a JavaScript function to check if a number is even."
        },
        {
            "role": "assistant",
            "content": """Here's a JavaScript function to check if a number is even:

```javascript
function isEven(number) {
    return number % 2 === 0;
}

// Test the function
console.log(isEven(4));  // true
console.log(isEven(7));  // false
```

This function returns true if the number is even and false if it's odd."""
        }
    ]
    
    expected_output = "true\nfalse"
    
    print("Running JavaScript code in E2B sandbox...")
    
    try:
        # Evaluate the code using E2B
        result = e2b_code_execution_reward(
            messages=messages,
            expected_output=expected_output,
            language="javascript",
            api_key=api_key,
            timeout=15  # Increase timeout for first-time sandbox creation
        )
        
        # Display results
        print(f"\nScore: {result.score:.2f}")
        print("\nMetrics:")
        
        for metric_name, metric in result.metrics.items():
            print(f"\n--- {metric_name} ---")
            print(metric.reason)
        
        # Verify success
        if result.score == 1.0:
            print("\n✅ E2B JavaScript integration test passed: Code executed successfully and outputs matched!")
        else:
            print("\n❌ E2B JavaScript integration test failed: Code executed but score was not 1.0")
            
    except Exception as e:
        print(f"\n❌ E2B JavaScript integration test failed with error: {str(e)}")

if __name__ == "__main__":
    main()