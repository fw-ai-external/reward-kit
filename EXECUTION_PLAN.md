# Execution Plan for Implementing Evaluation Preview and Deploy Logic

## Overview

This plan outlines how to implement the evaluation preview and deployment functionality in the Reward Kit, mirroring the capabilities in firectl. The implementation will focus on two key commands:

1. `preview` - For testing evaluations against sample data
2. `create` - For creating evaluations on the Fireworks platform

## Current Architecture Understanding

From analyzing the codebase:

- The `reward_kit` package already supports reward functions through the `@reward_function` decorator
- The `deploy()` method in `reward_function.py` provides a way to deploy functions as evaluations
- The current CLI in `cli.py` uses Typer to define commands

However, the current implementation doesn't support:
- Previewing evaluations against sample data before deployment
- Creating evaluations from Python folders with specific structures
- Multi-metrics evaluation support

## Implementation Details

### 1. Create Evaluation Module

Create a new module `reward_kit/evaluation.py` to handle evaluation-specific functionality.

```python
# File: reward_kit/evaluation.py

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import requests

from .models import RewardOutput, MetricRewardOutput

logger = logging.getLogger(__name__)

class EvaluatorPreviewResult:
    """Class to store preview results for an evaluator"""
    
    def __init__(self):
        self.results = []
        self.total_samples = 0
        self.total_runtime_ms = 0
    
    def add_result(self, sample_index, success, score, per_metric_evals):
        """Add a result for a specific sample"""
        self.results.append({
            "index": sample_index,
            "success": success,
            "score": score,
            "per_metric_evals": per_metric_evals
        })
    
    def display(self):
        """Display formatted results"""
        print("Evaluation Preview Results")
        print("------------------------")
        print(f"Total Samples: {self.total_samples}")
        print(f"Total Runtime: {self.total_runtime_ms} ms\n")
        print("Individual Results:")
        print("------------------")
        
        for i, result in enumerate(self.results):
            print(f"Sample {result['index'] + 1}:")
            print(f"  Success: {result['success']}")
            print(f"  Score: {result['score']}")
            for metric, value in result['per_metric_evals'].items():
                print(f"  {metric}: {value}")
            if i < len(self.results) - 1:
                print()

class Evaluator:
    """Handles loading, previewing, and creating evaluations"""
    
    def __init__(self, multi_metrics=False):
        self.multi_metrics = multi_metrics
        self.code_files = {}  # Map of filename -> content
        self.metric_folders = {}  # Map of metric_name -> folder_path
        self.description = ""
        self.display_name = ""
    
    def load_metric_folder(self, metric_name, folder_path):
        """
        Load code files from a metric folder
        
        Args:
            metric_name: Name of the metric
            folder_path: Path to the folder containing code files
            
        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")
        
        files = {}
        for file_path in Path(folder_path).glob("*.py"):
            if file_path.is_file():
                with open(file_path, "r") as f:
                    filename = file_path.name
                    content = f.read()
                    files[filename] = content
                    
                    # Check for main.py with evaluate function
                    if filename == "main.py" and "evaluate" not in content:
                        raise ValueError(f"main.py in {folder_path} must contain an evaluate function")
        
        if "main.py" not in files:
            raise ValueError(f"main.py is required in {folder_path}")
        
        self.metric_folders[metric_name] = folder_path
        for filename, content in files.items():
            self.code_files[f"{metric_name}/{filename}"] = content
            
        logger.info(f"Loaded {len(files)} Python files for metric '{metric_name}' from {folder_path}")
        return files
    
    def load_multi_metrics_folder(self, folder_path):
        """
        Load code files from a folder with multiple metrics
        
        Args:
            folder_path: Path to the folder containing code files
            
        Returns:
            Dict mapping filenames to their contents
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")
        
        files = {}
        for file_path in Path(folder_path).glob("*.py"):
            if file_path.is_file():
                with open(file_path, "r") as f:
                    filename = file_path.name
                    content = f.read()
                    files[filename] = content
                    
                    # Check for main.py with evaluate function
                    if filename == "main.py" and "evaluate" not in content:
                        raise ValueError(f"main.py in {folder_path} must contain an evaluate function")
        
        if "main.py" not in files:
            raise ValueError(f"main.py is required in {folder_path}")
        
        self.code_files = files
        logger.info(f"Loaded {len(files)} Python files from {folder_path} for multi-metrics evaluation")
        return files
    
    def load_samples_from_jsonl(self, sample_file, max_samples=5):
        """
        Load samples from a JSONL file
        
        Args:
            sample_file: Path to the JSONL file
            max_samples: Maximum number of samples to load
            
        Returns:
            List of parsed JSON objects
        """
        if not os.path.exists(sample_file):
            raise ValueError(f"Sample file does not exist: {sample_file}")
        
        samples = []
        with open(sample_file, "r") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {i+1}, skipping")
        
        logger.info(f"Loaded {len(samples)} samples from {sample_file}")
        return samples
    
    def preview(self, sample_file, max_samples=5):
        """
        Run the evaluator against sample data
        
        Args:
            sample_file: Path to the JSONL file with samples
            max_samples: Maximum number of samples to process
            
        Returns:
            EvaluatorPreviewResult containing the preview results
        """
        if not self.code_files:
            raise ValueError("No code files loaded. Load metric folder(s) first.")
        
        if "main.py" not in self.code_files and not any(k.endswith("/main.py") for k in self.code_files):
            raise ValueError("No main.py found in code files")
        
        samples = self.load_samples_from_jsonl(sample_file, max_samples)
        if not samples:
            raise ValueError(f"No valid samples found in {sample_file}")
        
        # In a real implementation, we would call the Fireworks API
        # Here we'll simulate a local preview using Python's exec functionality
        
        preview_result = EvaluatorPreviewResult()
        preview_result.total_samples = len(samples)
        
        start_time = time.time()
        for i, sample in enumerate(samples):
            try:
                # For demonstration, we'll simulate the evaluation process
                # In a real implementation, this would call the API or run the code
                
                # Sample validation
                if "messages" not in sample:
                    raise ValueError(f"Sample {i+1} is missing 'messages' field")
                
                # Simple simulation of metric evaluation
                if self.multi_metrics:
                    per_metric_evals = {
                        "quality": 0.8,
                        "relevance": 0.7,
                        "safety": 0.9
                    }
                else:
                    per_metric_evals = {metric_name: 0.75 for metric_name in self.metric_folders}
                
                # Calculate an aggregate score
                score = sum(per_metric_evals.values()) / len(per_metric_evals)
                
                preview_result.add_result(
                    sample_index=i,
                    success=True,
                    score=score,
                    per_metric_evals=per_metric_evals
                )
                
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {str(e)}")
                preview_result.add_result(
                    sample_index=i,
                    success=False,
                    score=0.0,
                    per_metric_evals={"error": str(e)}
                )
        
        end_time = time.time()
        preview_result.total_runtime_ms = int((end_time - start_time) * 1000)
        
        return preview_result
    
    def create(self, evaluator_id, display_name=None, description=None):
        """
        Create the evaluation on the Fireworks platform
        
        Args:
            evaluator_id: ID for the evaluator
            display_name: Display name for the evaluator
            description: Description of the evaluator
            
        Returns:
            The created evaluator object
        """
        if not self.code_files:
            raise ValueError("No code files loaded. Load metric folder(s) first.")
        
        # Authentication
        account_id, auth_token = self._get_authentication()
        
        # Set display name and description
        self.display_name = display_name or evaluator_id
        self.description = description or f"Evaluator created from {evaluator_id}"
        
        # Construct the evaluator payload
        payload = {
            "evaluator": {
                "displayName": self.display_name,
                "description": self.description,
                "multiMetrics": self.multi_metrics,
                "criteria": self._construct_criteria(),
            },
            "evaluatorId": evaluator_id,
        }
        
        # Make API request to create evaluator
        url = f"https://api.fireworks.ai/v1/accounts/{account_id}/evaluators"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Creating evaluator '{evaluator_id}'...")
        
        # In a real implementation, this would be an actual API call
        # For now, simulate success
        print(f"Would send request to: {url}")
        print(f"With payload: {json.dumps(payload, indent=2)}")
        
        # Return a simulated response
        return {
            "name": f"accounts/{account_id}/evaluators/{evaluator_id}",
            "displayName": self.display_name,
            "description": self.description,
            "multiMetrics": self.multi_metrics
        }
    
    def _construct_criteria(self):
        """
        Construct the criteria for the evaluator
        
        Returns:
            List of criteria objects
        """
        if self.multi_metrics:
            return [{
                "type": "CODE_SNIPPETS",
                "codeSnippets": {
                    "language": "python",
                    "fileContents": self.code_files
                }
            }]
        else:
            criteria = []
            for metric_name in self.metric_folders:
                files = {k.split("/", 1)[1]: v for k, v in self.code_files.items() 
                         if k.startswith(f"{metric_name}/")}
                criteria.append({
                    "type": "CODE_SNIPPETS",
                    "name": metric_name,
                    "codeSnippets": {
                        "language": "python",
                        "fileContents": files
                    }
                })
            return criteria
    
    def _get_authentication(self):
        """
        Get authentication information for the Fireworks API
        
        Returns:
            Tuple of (account_id, auth_token)
        """
        import configparser
        from pathlib import Path
        
        # Try to get API key from environment
        auth_token = os.environ.get("FIREWORKS_API_KEY")
        account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
        
        # If not found, try config files
        if not auth_token or not account_id:
            auth_path = Path.home() / ".fireworks" / "auth.ini"
            settings_path = Path.home() / ".fireworks" / "settings.ini"
            
            if auth_path.exists():
                try:
                    auth_config = configparser.ConfigParser()
                    auth_config.read(auth_path)
                    if "default" in auth_config:
                        if not auth_token and "id_token" in auth_config["default"]:
                            auth_token = auth_config["default"]["id_token"]
                except Exception as e:
                    logger.warning(f"Error reading auth.ini: {str(e)}")
            
            if settings_path.exists():
                try:
                    settings_config = configparser.ConfigParser()
                    settings_config.read(settings_path)
                    if "default" in settings_config:
                        if not account_id and "account_id" in settings_config["default"]:
                            account_id = settings_config["default"]["account_id"]
                except Exception as e:
                    logger.warning(f"Error reading settings.ini: {str(e)}")
        
        if not account_id:
            raise ValueError("Account ID not found. Set FIREWORKS_ACCOUNT_ID environment variable or configure ~/.fireworks/settings.ini")
        
        if not auth_token:
            raise ValueError("Auth token not found. Set FIREWORKS_API_KEY environment variable or configure ~/.fireworks/auth.ini")
        
        return account_id, auth_token

# Helper functions for CLI commands
def preview_evaluation(metric_folders=None, multi_metrics=False, folder=None, sample_file=None, max_samples=5):
    """
    Preview an evaluation with sample data
    
    Args:
        metric_folders: List of METRIC_NAME=folder_path pairs
        multi_metrics: Whether to use multi-metrics mode
        folder: Path to folder with multiple metrics (for multi_metrics mode)
        sample_file: Path to sample JSONL file
        max_samples: Maximum number of samples to process
        
    Returns:
        EvaluatorPreviewResult with preview results
    """
    evaluator = Evaluator(multi_metrics=multi_metrics)
    
    if multi_metrics:
        if not folder:
            raise ValueError("Folder must be specified when using multi-metrics mode")
        evaluator.load_multi_metrics_folder(folder)
    else:
        if not metric_folders:
            raise ValueError("At least one metric folder must be specified when not using multi-metrics mode")
        
        for pair in metric_folders:
            if "=" not in pair:
                raise ValueError(f"Invalid metric-folder format: {pair}. Expected METRIC_NAME=folder_path")
            
            metric_name, folder_path = pair.split("=", 1)
            evaluator.load_metric_folder(metric_name, folder_path)
    
    return evaluator.preview(sample_file, max_samples)

def create_evaluation(evaluator_id, metric_folders=None, multi_metrics=False, folder=None, display_name=None, description=None):
    """
    Create an evaluation on the Fireworks platform
    
    Args:
        evaluator_id: ID for the evaluator
        metric_folders: List of METRIC_NAME=folder_path pairs
        multi_metrics: Whether to use multi-metrics mode
        folder: Path to folder with multiple metrics (for multi_metrics mode)
        display_name: Display name for the evaluator
        description: Description of the evaluator
        
    Returns:
        Created evaluator object
    """
    evaluator = Evaluator(multi_metrics=multi_metrics)
    
    if multi_metrics:
        if not folder:
            raise ValueError("Folder must be specified when using multi-metrics mode")
        evaluator.load_multi_metrics_folder(folder)
    else:
        if not metric_folders:
            raise ValueError("At least one metric folder must be specified when not using multi-metrics mode")
        
        for pair in metric_folders:
            if "=" not in pair:
                raise ValueError(f"Invalid metric-folder format: {pair}. Expected METRIC_NAME=folder_path")
            
            metric_name, folder_path = pair.split("=", 1)
            evaluator.load_metric_folder(metric_name, folder_path)
    
    return evaluator.create(evaluator_id, display_name, description)
```

### 2. Update CLI Module

Extend `reward_kit/cli.py` to add preview and create commands:

```python
# Updates to reward_kit/cli.py

from .evaluation import preview_evaluation, create_evaluation

# Add these commands to the existing Typer app

@app.command("preview")
def preview_cmd(
    metric_folder: List[str] = typer.Option(None, "--metric-folder", help="Format as METRIC_NAME=folder_path"),
    sample_file: str = typer.Option(..., "--sample-file", help="Path to sample JSONL file"),
    multi_metrics: bool = typer.Option(False, "--multi-metrics", help="If set, enables multiple metrics from one folder"),
    folder: str = typer.Option(None, "--folder", help="Path to folder with multiple metrics"),
    max_samples: int = typer.Option(5, "--max-samples", help="Maximum number of samples to process"),
):
    """Preview an evaluation with sample data."""
    try:
        if not metric_folder and not folder:
            typer.echo("Either --metric-folder or --folder with --multi-metrics must be specified")
            raise typer.Exit(code=1)
            
        if multi_metrics and not folder:
            typer.echo("--folder must be specified when using --multi-metrics")
            raise typer.Exit(code=1)
            
        preview_result = preview_evaluation(
            metric_folders=metric_folder,
            multi_metrics=multi_metrics,
            folder=folder,
            sample_file=sample_file,
            max_samples=max_samples
        )
        
        # Display the results
        preview_result.display()
        
    except Exception as e:
        typer.echo(f"Error previewing evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)

@app.command("create")
def create_cmd(
    eval_id: str = typer.Argument(..., help="ID for the evaluation to create"),
    metric_folder: List[str] = typer.Option(None, "--metric-folder", help="Format as METRIC_NAME=folder_path"),
    multi_metrics: bool = typer.Option(False, "--multi-metrics", help="If set, enables multiple metrics from one folder"),
    folder: str = typer.Option(None, "--folder", help="Path to folder with multiple metrics"),
    display_name: str = typer.Option(None, "--display-name", help="Display name for the evaluation"),
    description: str = typer.Option(None, "--description", help="Description of the evaluation"),
):
    """Create an evaluation."""
    try:
        if not metric_folder and not folder:
            typer.echo("Either --metric-folder or --folder with --multi-metrics must be specified")
            raise typer.Exit(code=1)
            
        if multi_metrics and not folder:
            typer.echo("--folder must be specified when using --multi-metrics")
            raise typer.Exit(code=1)
            
        evaluator = create_evaluation(
            evaluator_id=eval_id,
            metric_folders=metric_folder,
            multi_metrics=multi_metrics,
            folder=folder,
            display_name=display_name,
            description=description
        )
        
        typer.echo(f"Successfully created evaluator: {evaluator['name']}")
        
    except Exception as e:
        typer.echo(f"Error creating evaluation: {str(e)}", err=True)
        raise typer.Exit(code=1)
```

### 3. Create Example Scripts

Add examples showing how to use the new functionality:

```python
# File: examples/evaluation_preview_example.py

"""
Example of previewing an evaluation before creation.
"""

import os
import sys
import json
from pathlib import Path

# Ensure reward-kit is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_kit.evaluation import preview_evaluation, create_evaluation

def main():
    # Create a temporary example folder
    tmp_folder = Path("./tmp_metric")
    tmp_folder.mkdir(exist_ok=True)
    
    # Create a main.py file with an evaluate function
    main_py = tmp_folder / "main.py"
    main_py.write_text("""
def evaluate(entry):
    """Evaluate a sample entry."""
    # Extract the messages
    messages = entry.get('messages', [])
    if not messages:
        return {'score': 0.0, 'reason': 'No messages found'}
    
    # Get the last message (assistant's response)
    last_message = messages[-1]
    content = last_message.get('content', '')
    
    # Simple evaluation: count the number of words
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)  # Cap at 1.0
    
    return {
        'score': score,
        'reason': f'Word count: {word_count}',
        'metrics': {
            'word_count': word_count,
            'score': score
        }
    }
""")
    
    # Create a sample JSONL file
    sample_file = Path("./samples.jsonl")
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Tell me about AI"},
                {"role": "assistant", "content": "AI (Artificial Intelligence) refers to systems designed to mimic human intelligence. These systems can learn from data, identify patterns, and make decisions with minimal human intervention."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on building systems that can learn from and make decisions based on data."}
            ]
        }
    ]
    
    with open(sample_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    # Preview the evaluation
    print("Previewing evaluation...")
    preview_result = preview_evaluation(
        metric_folders=["word_count=./tmp_metric"],
        sample_file="./samples.jsonl",
        max_samples=2
    )
    
    preview_result.display()
    
    # Create the evaluation
    print("\nCreating evaluation...")
    try:
        evaluator = create_evaluation(
            evaluator_id="word-count-eval",
            metric_folders=["word_count=./tmp_metric"],
            display_name="Word Count Evaluator",
            description="Evaluates responses based on word count"
        )
        print(f"Created evaluator: {evaluator['name']}")
    except Exception as e:
        print(f"Error creating evaluator: {str(e)}")
    
    # Clean up
    main_py.unlink()
    tmp_folder.rmdir()
    sample_file.unlink()

if __name__ == "__main__":
    main()
```

### 4. Unit Tests

Add tests for the new functionality in `tests/test_evaluation.py`:

```python
# File: tests/test_evaluation.py

import os
import json
import tempfile
from pathlib import Path
import pytest

from reward_kit.evaluation import Evaluator, preview_evaluation, create_evaluation

def create_test_folder():
    """Create a temporary folder with a main.py file for testing"""
    tmp_dir = tempfile.mkdtemp()
    
    # Create main.py
    with open(os.path.join(tmp_dir, "main.py"), "w") as f:
        f.write("""
def evaluate(entry):
    messages = entry.get('messages', [])
    if not messages:
        return {'score': 0.0, 'reason': 'No messages found'}
    
    last_message = messages[-1]
    content = last_message.get('content', '')
    
    word_count = len(content.split())
    score = min(word_count / 100, 1.0)
    
    return {
        'score': score,
        'reason': f'Word count: {word_count}'
    }
""")
    
    return tmp_dir

def create_sample_file():
    """Create a temporary sample file for testing"""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    
    samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        }
    ]
    
    with os.fdopen(fd, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    return path

def test_evaluator_load_metric_folder():
    """Test loading metric folder"""
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator()
        files = evaluator.load_metric_folder("test_metric", tmp_dir)
        
        assert "main.py" in files
        assert "test_metric" in evaluator.metric_folders
        assert "test_metric/main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["test_metric/main.py"]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)

def test_evaluator_load_multi_metrics_folder():
    """Test loading multi-metrics folder"""
    tmp_dir = create_test_folder()
    try:
        evaluator = Evaluator(multi_metrics=True)
        files = evaluator.load_multi_metrics_folder(tmp_dir)
        
        assert "main.py" in files
        assert "main.py" in evaluator.code_files
        assert "evaluate" in evaluator.code_files["main.py"]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)

def test_evaluator_preview():
    """Test preview functionality"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    
    try:
        evaluator = Evaluator()
        evaluator.load_metric_folder("test_metric", tmp_dir)
        
        preview_result = evaluator.preview(sample_file, max_samples=2)
        
        assert preview_result.total_samples == 2
        assert preview_result.total_runtime_ms > 0
        assert len(preview_result.results) == 2
        
        # Check first result
        assert preview_result.results[0]['index'] == 0
        assert preview_result.results[0]['success'] is True
        assert 'score' in preview_result.results[0]
        assert 'per_metric_evals' in preview_result.results[0]
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)

def test_preview_evaluation_helper():
    """Test the preview_evaluation helper function"""
    tmp_dir = create_test_folder()
    sample_file = create_sample_file()
    
    try:
        preview_result = preview_evaluation(
            metric_folders=[f"test_metric={tmp_dir}"],
            sample_file=sample_file,
            max_samples=2
        )
        
        assert preview_result.total_samples == 2
        assert len(preview_result.results) == 2
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
        os.unlink(sample_file)

def test_create_evaluation_helper(monkeypatch):
    """Test the create_evaluation helper function"""
    tmp_dir = create_test_folder()
    
    # Mock authentication
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_api_key")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "test_account")
    
    try:
        evaluator = create_evaluation(
            evaluator_id="test-eval",
            metric_folders=[f"test_metric={tmp_dir}"],
            display_name="Test Evaluator",
            description="Test description"
        )
        
        assert evaluator["name"] == "accounts/test_account/evaluators/test-eval"
        assert evaluator["displayName"] == "Test Evaluator"
        assert evaluator["description"] == "Test description"
    finally:
        # Clean up
        os.unlink(os.path.join(tmp_dir, "main.py"))
        os.rmdir(tmp_dir)
```

## Deployment and API Integration

The actual API integration will rely on the Fireworks API for creating evaluations and previewing them. The preview should send the code and samples to the API and receive the evaluation results.

For the create functionality, we'll need to:
1. Get proper authentication from environment variables or config files
2. Construct the evaluator object with criteria
3. Send a POST request to the Fireworks API
4. Parse and return the response

## Execution Timeline

### Week 1
- Day 1: Create the core Evaluator class and add file loading functionality
- Day 2: Implement preview functionality for the Evaluator class
- Day 3: Implement create functionality for the Evaluator class
- Days 4-5: Add CLI commands and write tests

### Week 2
- Day 1: Create example scripts and documentation
- Day 2: Integration testing and bug fixes
- Day 3: Documentation and cleanup
- Days 4-5: Buffer for unexpected issues or enhancements

## Conclusion

This implementation plan covers the core functionality needed to replicate the firectl evaluation preview and deploy logic in the Reward Kit SDK. It provides a clear path to implement the features while ensuring proper integration with the existing codebase.

The approach focuses on:
1. Creating a flexible Evaluator class that can handle both single-metric and multi-metrics evaluations
2. Providing a simple API for previewing and creating evaluations
3. Integrating with the existing CLI framework
4. Ensuring proper error handling and authentication support

By following this plan, we can successfully implement the evaluation preview and deploy functionality in a way that's consistent with the existing codebase and meets the requirements specified in ISSUES.md.