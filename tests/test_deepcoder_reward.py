import unittest
import json
import os
from typing import List, Dict, Any

from reward_kit.rewards import deepcoder_code_reward
from reward_kit.models import Message, EvaluateResult

# Path to the transformed sample data
TRANSFORMED_SAMPLE_DATA_PATH = "examples/trl_integration/data/deepcoder_mvp_transformed_sample.jsonl"

# Check if E2B API key is available for E2B tests
E2B_API_KEY = os.environ.get("E2B_API_KEY")
E2B_AVAILABLE = bool(E2B_API_KEY)

class TestDeepCoderReward(unittest.TestCase):
    SAMPLES: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls):
        """Load the transformed sample data once for all tests."""
        try:
            with open(TRANSFORMED_SAMPLE_DATA_PATH, 'r') as f:
                for line in f:
                    cls.SAMPLES.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Warning: Test data file not found at {TRANSFORMED_SAMPLE_DATA_PATH}")
            # Allow tests to run but they might fail or be skipped if they depend on this data.
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {TRANSFORMED_SAMPLE_DATA_PATH}")


    def test_python_all_tests_pass_local(self):
        """Test Python code that passes all test cases locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        
        sample = self.SAMPLES[0] # add_one function
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\ndef add_one(x):\n  return int(x) + 1\n\n# To be called by the testing harness\nval = input()\nprint(add_one(val))\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertIn("test_results", result.metrics)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(all(tc.get("passed") for tc in details))
        # Dictionary access
        self.assertEqual(result['score'], 1.0)
        self.assertIn("test_results", result['metrics'])
        if "test_results" in result['metrics'] and result['metrics']["test_results"]['reason']:
            details = json.loads(result['metrics']["test_results"]['reason'])
            self.assertTrue(all(tc.get("passed") for tc in details))

    def test_python_one_test_fails_local(self):
        """Test Python code where one test case fails locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0] # add_one function, but we make it add_two
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\ndef add_one(x):\n  return int(x) + 2\n\nval = input()\nprint(add_one(val))\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertFalse(details[0].get("passed")) # First test case (5 -> expected 6, actual 7) should fail
        # Dictionary access
        self.assertEqual(result['score'], 0.0)
        if "test_results" in result['metrics'] and result['metrics']["test_results"]['reason']:
            details = json.loads(result['metrics']["test_results"]['reason'])
            self.assertFalse(details[0].get("passed"))

    def test_python_syntax_error_local(self):
        """Test Python code with a syntax error locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\ndef add_one(x)\n  return x + 1\n\nval = input()\nprint(add_one(val))\n```") # Missing colon
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(any("error" in tc for tc in details))
        # Dictionary access
        self.assertEqual(result['score'], 0.0)
        if "test_results" in result['metrics'] and result['metrics']["test_results"]['reason']:
            details = json.loads(result['metrics']["test_results"]['reason'])
            self.assertTrue(any("error" in tc for tc in details))


    def test_python_timeout_local(self):
        """Test Python code that times out locally."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\nimport time\ndef add_one(x):\n  time.sleep(15)\n  return int(x) + 1\n\nval = input()\nprint(add_one(val))\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            timeout=2, # Short timeout
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(any(tc.get("error") and "timed out" in str(tc.get("error")).lower() for tc in details))
        # Dictionary access
        self.assertEqual(result['score'], 0.0)
        if "test_results" in result['metrics'] and result['metrics']["test_results"]['reason']:
            details = json.loads(result['metrics']["test_results"]['reason'])
            self.assertTrue(any(tc.get("error") and "timed out" in str(tc.get("error")).lower() for tc in details))

    def test_no_code_block(self):
        """Test when no code block is found in the assistant message."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        sample = self.SAMPLES[0]
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="I am not sure how to do that.")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metrics)
        if "error" in result.metrics:
             self.assertIn("No python code block found", result.metrics["error"].reason)
        # Dictionary access
        self.assertEqual(result['score'], 0.0)
        self.assertIn("error", result['metrics'])
        if "error" in result['metrics']:
             self.assertIn("No python code block found", result['metrics']["error"]['reason'])

    def test_javascript_all_tests_pass_local(self):
        """Test JavaScript code that passes all test cases locally."""
        # Using a simplified version of add_one for JS
        js_test_cases = [{"input": "5", "expected_output": "6\n"}, {"input": "-2", "expected_output": "-1\n"}]
        messages = [
            Message(role="user", content="Write a JS function addOne."),
            Message(role="assistant", content="```javascript\nfunction addOne(x) {\n  return parseInt(x) + 1;\n}\n\nconst val = readline();\nconsole.log(addOne(val));\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="javascript",
            test_cases=js_test_cases,
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result['score'], 1.0)

    def test_javascript_one_test_fails_local(self):
        """Test JavaScript code where one test case fails locally."""
        js_test_cases = [{"input": "5", "expected_output": "6\n"}, {"input": "-2", "expected_output": "-1\n"}]
        messages = [
            Message(role="user", content="Write a JS function addOne."),
            Message(role="assistant", content="```javascript\nfunction addOne(x) {\n  return parseInt(x) + 2; // Bug here\n}\n\nconst val = readline();\nconsole.log(addOne(val));\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="javascript",
            test_cases=js_test_cases,
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result['score'], 0.0)

    @unittest.skip("Skipping E2B tests due to intermittent service issues")
    @unittest.skipUnless(E2B_AVAILABLE, "E2B_API_KEY not set, skipping E2B tests.")
    def test_python_all_tests_pass_e2b(self):
        """Test Python code that passes all test cases in E2B."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")
        
        sample = self.SAMPLES[0] # add_one function
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\ndef add_one(x):\n  return int(x) + 1\n\nval = input()\nprint(add_one(val))\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="e2b",
            api_key=E2B_API_KEY 
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 1.0)
        self.assertIn("test_results", result.metrics)
        if "test_results" in result.metrics and result.metrics["test_results"].reason:
            details = json.loads(result.metrics["test_results"].reason)
            self.assertTrue(all(tc.get("passed") for tc in details))
        # Dictionary access
        self.assertEqual(result['score'], 1.0)
        self.assertIn("test_results", result['metrics'])
        if "test_results" in result['metrics'] and result['metrics']["test_results"]['reason']:
            details = json.loads(result['metrics']["test_results"]['reason'])
            self.assertTrue(all(tc.get("passed") for tc in details))

    @unittest.skip("Skipping E2B tests due to intermittent service issues")
    @unittest.skipUnless(E2B_AVAILABLE, "E2B_API_KEY not set, skipping E2B tests.")
    def test_python_one_test_fails_e2b(self):
        """Test Python code where one test case fails in E2B."""
        if not self.SAMPLES:
            self.skipTest("Test data not loaded.")

        sample = self.SAMPLES[0] # add_one function, but we make it add_two
        messages = [
            Message(role="user", content=sample["prompt"]),
            Message(role="assistant", content="```python\ndef add_one(x):\n  return int(x) + 2\n\nval = input()\nprint(add_one(val))\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=sample["test_cases"],
            environment="e2b",
            api_key=E2B_API_KEY
        )
        self.assertIsInstance(result, EvaluateResult)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result['score'], 0.0)

    def test_empty_test_cases(self):
        """Test behavior with an empty list of test cases."""
        messages = [
            Message(role="user", content="Prompt"),
            Message(role="assistant", content="```python\nprint('hello')\n```")
        ]
        result = deepcoder_code_reward(
            messages=messages,
            language="python",
            test_cases=[], # Empty test cases
            environment="local"
        )
        self.assertIsInstance(result, EvaluateResult)
        # Attribute access
        self.assertEqual(result.score, 0.0) # Should be 0 as per implementation
        self.assertIn("error", result.metrics)
        if "error" in result.metrics:
            self.assertEqual(result.metrics["error"].reason, "No test cases provided.")
        # Dictionary access
        self.assertEqual(result['score'], 0.0)
        self.assertIn("error", result['metrics'])
        if "error" in result['metrics']:
            self.assertEqual(result['metrics']["error"]['reason'], "No test cases provided.")

if __name__ == '__main__':
    unittest.main()
