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

#### 1.2 API Reference - COMPLETED
- **Core Classes and Methods**: Full reference for `RewardFunction`, `reward_function`, etc. ✅
- **Data Models**: Reference for all data models (`RewardOutput`, `Message`, etc.) ✅
- **Deployment Methods**: Reference for deployment-related functions ✅

#### 1.3 Code Examples - COMPLETED
- **Basic Reward Function**: Simple examples like word count ✅
- **Advanced Reward Functions**: Multiple metrics, combining metrics ✅
- **Function Calling Evaluation**: Specialized reward functions for evaluating tool use ✅

### 2. Tutorials and Guides

#### 2.1 Step-by-Step Tutorials - COMPLETED
- **Creating Your First Reward Function**: From scratch to deployment ✅
- **Evaluating Model Responses**: Using reward functions for evaluation ✅
- **Integrating with Training Workflows**: How to use reward functions in RLHF

#### 2.2 Best Practices - COMPLETED
- **Designing Effective Reward Functions**: Guidelines and principles ✅
- **Handling Edge Cases**: Ensuring robust evaluation ✅
- **Performance Optimization**: Making reward functions efficient ✅

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

#### 4.1 CLI Documentation - COMPLETED
- **Command Reference**: All CLI commands with examples ✅
- **Workflow Integration**: Using CLI in development workflows ✅

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

## Function Calling Reward Implementation Plan

### Goal
Create a comprehensive function calling evaluation system that combines schema validation and LLM evaluation to assess the quality of function calls made by AI models.

### Components

#### 1. Schema Jaccard Distance Reward
- Implement a JSON schema validation reward function that:
  - Takes a function call output and expected schema as input
  - Validates function name matches
  - Calculates Jaccard similarity between expected and actual schema
  - Returns a score between 0.0-1.0 based on schema similarity
  - Provides detailed explanations for mismatches

#### 2. LLM Judge Reward
- Implement an LLM-based judge reward function that:
  - Takes a function call output and expected behavior
  - Sends to GPT-4o-mini for evaluation
  - Returns a score based on LLM's judgment of correctness
  - Provides qualitative feedback on function call quality

#### 3. Composite Function Calling Reward
- Implement a composite reward that:
  - Combines schema validation and LLM judgment
  - Allows configurable weights for each component
  - Returns detailed metrics for debugging and analysis

### Implementation Steps

1. Schema Jaccard Reward:
   - Create schema validation utilities
   - Implement Jaccard similarity calculation
   - Add proper error handling and edge cases
   - Create unit tests

2. LLM Judge Reward:
   - Set up GPT-4o-mini API integration
   - Design effective prompt templates
   - Handle API responses and error cases
   - Create unit tests

3. Composite Reward:
   - Implement combined reward function
   - Add configurable weighting
   - Create comprehensive documentation
   - Create integration tests

4. Documentation and Examples:
   - Add function calling examples
   - Create tutorial for function call evaluation
   - Update API reference

### Testing Approach
- Unit tests for each individual component
- Integration tests for composite reward
- Example notebooks with real-world scenarios
- Benchmarking against human judgments

## Documentation Status

The core developer guide and basic examples are now complete. These provide a solid foundation for developers to understand the basics of creating, testing, and deploying reward functions.

### Completed Items

1. ✅ Implement Schema Jaccard Distance Reward for function calling
2. ✅ Implement LLM Judge Reward using GPT-4o-mini
3. ✅ Create Composite Function Calling Reward
4. ✅ Add documentation and examples for function calling rewards

### Next Priority Items

1. Add advanced topics documentation
2. Create deployment guide documentation
3. Develop reference implementation templates
4. Create practical examples and cookbooks

The documentation can be found in the `docs/` directory with the following structure:
- `docs/developer_guide/`: Core concepts and usage guides
- `docs/examples/`: Code examples
- `docs/tutorials/`: Step-by-step guides
- `docs/api_reference/`: API documentation
- `docs/cli_reference/`: CLI documentation
- `docs/DOCUMENTATION_STATUS.md`: Detailed status and recommendations