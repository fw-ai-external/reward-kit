# Reward Kit Documentation and Implementation Improvements

## Addressing User Feedback

The following changes have been made to address feedback from users:

```
Not sure I have the full context but looks quite useful for creating complex reward functions
A few questions:
- What is the reward-kit CLI and how is it integrated with FW?
- Is Reward Kit going to be the only way to create evaluators?
- Confused b/w Evaluation Types and Reward Output Types. Examples show output of evaluator as a RewardOutput type, but evaluation types section mentions EvaluateResult as the FW compatible output type
```

## Changes Made

### 1. Simplified Type System: Standardized on EvaluateResult

To address the confusion between RewardOutput and EvaluateResult, we have:

- Deprecated RewardOutput and standardized on EvaluateResult
- Added conversion methods between the two types
- Updated reward functions to return EvaluateResult
- Added appropriate deprecation warnings

This simplifies the API by having a single output type (EvaluateResult) instead of two separate types.

### 2. Reward Kit CLI and Fireworks Integration

The Reward Kit Command Line Interface (CLI) is a command-line tool that provides utilities for working with reward functions and evaluators. It integrates with Fireworks (FW) in the following ways:

- **Authentication**: Uses Fireworks API keys for authentication through environment variables (`FIREWORKS_API_KEY`)
- **Deployment**: Deploys reward functions to Fireworks as evaluators via the `deploy` command
- **Preview**: Tests evaluations locally before deployment with the `preview` command
- **Agent Evaluation**: Runs agent evaluations using the Fireworks API via the `agent-eval` command

The CLI serves as a bridge between local development and the Fireworks platform, simplifying the deployment and management of evaluators.

See the [CLI Reference](docs/cli_reference/cli_overview.mdx) for full details.

### 3. Methods for Creating Evaluators

Reward Kit is NOT the only way to create evaluators on Fireworks, but it provides a streamlined approach. Evaluators can be created through:

1. **Reward Kit**: The recommended approach for Python-based evaluators, using the `@reward_function` decorator and `.deploy()` method
2. **Direct API**: Using the Fireworks REST API to create evaluators manually
3. **Fireworks Console**: Creating evaluators through the web-based Fireworks console (UI)

Reward Kit simplifies the process by handling the deployment process, standardizing inputs/outputs, and providing local testing capabilities before deployment.

## Implementation Updates

### 1. Deprecation of RewardOutput

The RewardOutput class has been deprecated in favor of EvaluateResult. It will be fully removed in a future version.

```python
# Old approach (deprecated)
@reward_function
def my_reward(messages, **kwargs) -> RewardOutput:
    return RewardOutput(score=0.8, metrics={...})

# New approach
@reward_function
def my_reward(messages, **kwargs) -> EvaluateResult:
    return EvaluateResult(score=0.8, reason="Good response", metrics={...})
```

### 2. Conversion Utilities

Added conversion methods between the two types:

```python
# Convert from RewardOutput to EvaluateResult
evaluate_result = reward_output.to_evaluate_result()

# Convert from EvaluateResult to RewardOutput (for backwards compatibility)
reward_output = evaluate_result.to_reward_output()
```

### 3. Backward Compatibility

We've maintained backward compatibility by:

- Adding type conversion methods
- Supporting both return types in the RewardFunction.__call__ method
- Adding deprecation warnings

## Documentation Updates

The documentation has been updated to clarify these points in the following places:

- CLI Reference: Added more details about Fireworks integration
- API Reference: Clarified that EvaluateResult is the preferred output type
- Developer Guide: Added a section on different methods for creating evaluators

## Current Status

All changes have been implemented and tested. The code now:

1. Uses EvaluateResult as the standard return type for reward functions
2. Provides deprecation warnings for RewardOutput
3. Includes conversion methods between types for backward compatibility
4. Updates documentation to reflect these changes

All tests are passing, though there are some type checking warnings that could be addressed in a follow-up update.