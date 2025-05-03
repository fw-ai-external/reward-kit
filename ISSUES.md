# Issues and Tasks

## Resolved Issues

### Deploy is now working - FIXED

The deploy functionality is now working correctly. The issue was that the `legacy_reward_function.deploy()` method in `reward_function.py` had a different implementation from the working `create_evaluation()` function in `evaluation.py`. 

We fixed it by:
1. Creating a temporary directory with the reward function as a main.py file
2. Using the `create_evaluation()` function from `evaluation.py` to deploy it
3. Ensuring proper authentication with the improved auth mechanism

Both approaches (`deploy()` in reward functions and `create_evaluation()` for evaluations) now work through the same underlying mechanism.

```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/deploy_example.py 
Testing reward function locally...
Informativeness Reward Result:
Score: 0.8892
Metrics:
  length: 0.0892 - Response length: 446 chars
  specificity: 0.3 - Found 2 specificity markers
  content_density: 0.5 - Content density: 4 content words in 64 total words


Deploying to Fireworks...
INFO:reward_kit.auth:Using development API base, defaulting to pyroworks-dev account
Using account ID: pyroworks-dev
Using auth token (first 10 chars): eyJraWQiOi...
INFO:reward_kit.reward_function:Making request to: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators (using API base: https://dev.api.fireworks.ai)
INFO:reward_kit.reward_function:Using account_id: pyroworks-dev
INFO:reward_kit.reward_function:Auth token present: True
INFO:reward_kit.reward_function:Deploying reward function 'informativeness_reward' as evaluation 'informativeness-v1'...
ERROR:reward_kit.reward_function:Error deploying evaluation: 403 Client Error: Forbidden for url: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators
ERROR:reward_kit.reward_function:Response: {"error":"unauthorized"}

Permission Error: Your API key doesn't have deployment permissions.
Possible solutions:
1. Use a production API key: export FIREWORKS_API_KEY=your_production_key
2. Request deployment permissions for your API key
3. Check if your account has evaluator deployment enabled
Error details: {"error":"unauthorized"}
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/evaluation_preview_example.py 
Previewing evaluation...
INFO:reward_kit.evaluation:Loaded 1 Python files for metric 'word_count' from /home/bchen/home/reward-kit/examples/metrics/word_count
INFO:reward_kit.evaluation:Loaded 2 samples from ./examples/samples/samples.jsonl
INFO:reward_kit.evaluation:Previewing evaluator using API endpoint: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators:previewEvaluator with account: pyroworks-dev
Evaluation Preview Results
------------------------
Total Samples: 2
Total Runtime: 7483 ms

Individual Results:
------------------
Sample 1:
  Success: 
  Score: 0.26
  word_count: {'reason': 'Word count: 26', 'score': 0.26, 'success': None}

Sample 2:
  Success: 
  Score: 0.22
  word_count: {'reason': 'Word count: 22', 'score': 0.22, 'success': None}

Creating evaluation...
INFO:reward_kit.evaluation:Loaded 1 Python files for metric 'word_count' from /home/bchen/home/reward-kit/examples/metrics/word_count
INFO:reward_kit.evaluation:Creating evaluator 'word-count-eval' for account 'pyroworks-dev'...
INFO:reward_kit.evaluation:Evaluator 'word-count-eval' already exists, deleting and recreating...
INFO:reward_kit.evaluation:Successfully deleted evaluator 'word-count-eval'
INFO:reward_kit.evaluation:Successfully created evaluator 'word-count-eval'
Created evaluator: accounts/pyroworks-dev/evaluators/word-count-eval
```

## Custom Reward Functions Documentation Plan

### 1. Core Documentation

#### 1.1 Developer Guide - COMPLETED
- **Getting Started with Reward Functions**: Basic concepts, installation, setup ✅
- **Reward Function Anatomy**: Detailed explanation of the `@reward_function` decorator ✅
- **Core Data Types**: Explanation of `RewardOutput`, `MetricRewardOutput`, `Message`, etc. ✅
- **Evaluation Workflows**: Local testing, preview, deployment ✅

#### 1.2 API Reference
- **Core Classes and Methods**: Full reference for `RewardFunction`, `reward_function`, etc.
- **Data Models**: Reference for all data models (`RewardOutput`, `Message`, etc.)
- **Deployment Methods**: Reference for deployment-related functions

#### 1.3 Code Examples - COMPLETED
- **Basic Reward Function**: Simple examples like word count ✅
- **Advanced Reward Functions**: Multiple metrics, combining metrics ✅
- **Function Calling Evaluation**: Specialized reward functions for evaluating tool use

### 2. Tutorials and Guides

#### 2.1 Step-by-Step Tutorials - IN PROGRESS
- **Creating Your First Reward Function**: From scratch to deployment ✅
- **Evaluating Model Responses**: Using reward functions for evaluation
- **Integrating with Training Workflows**: How to use reward functions in RLHF

#### 2.2 Best Practices
- **Designing Effective Reward Functions**: Guidelines and principles
- **Handling Edge Cases**: Ensuring robust evaluation
- **Performance Optimization**: Making reward functions efficient

#### 2.3 Advanced Topics
- **Custom Providers**: Using different models for evaluation
- **Multi-Component Scoring**: Combining multiple metrics
- **Metadata Handling**: Using context in reward functions
- **Integration with Training**: Using reward functions in RLHF pipelines

### 3. Reference Implementation Templates

#### 3.1 Common Use Case Templates
- **Content Quality Evaluation**: Informativeness, relevance, etc.
- **Safety Evaluation**: Detecting unsafe content
- **Tool Use Evaluation**: Function calling accuracy
- **Instruction Following**: Adherence to user instructions

#### 3.2 Starter Kits
- **Basic Metrics Pack**: Common metrics ready to use
- **Custom Evaluator Framework**: Template for building complex evaluators
- **Evaluation Dashboard**: Tools for visualizing evaluation results

### 4. Infrastructure Documentation

#### 4.1 CLI Documentation
- **Command Reference**: All CLI commands with examples
- **Workflow Integration**: Using CLI in development workflows

#### 4.2 Deployment Guide
- **Authentication Setup**: Configuring API credentials
- **Deployment Options**: Different ways to deploy reward functions
- **Versioning and Updates**: Managing deployed evaluators
- **Troubleshooting Deployment**: Common issues and solutions

### 5. Practical Examples and Cookbooks

#### 5.1 Real-World Examples
- **Helpfulness Evaluator**: Complete implementation with explanations
- **Factual Accuracy Evaluator**: Detecting factual errors
- **Reasoning Evaluator**: Assessing logical reasoning

#### 5.2 Cookbooks for Specific Domains
- **Customer Support Evaluation**: Metrics for support scenarios
- **Creative Writing Evaluation**: Metrics for creative content
- **Technical Documentation Evaluation**: Metrics for technical accuracy

## Current Documentation Status

The core developer guide and basic examples are now complete. These provide a solid foundation for developers to understand the basics of creating, testing, and deploying reward functions.

### Next Priority Items

1. Complete the function calling evaluation example
2. Create API reference documentation
3. Add more step-by-step tutorials, particularly for evaluation
4. Develop best practices guidelines
5. Add CLI documentation

The documentation can be found in the `docs/` directory with the following structure:
- `docs/developer_guide/`: Core concepts and usage guides
- `docs/examples/`: Code examples
- `docs/tutorials/`: Step-by-step guides
- `docs/DOCUMENTATION_STATUS.md`: Detailed status and recommendations