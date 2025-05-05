# Next set of ticket items
## TRL adapter for reward functions

We want to make sure the reward functions can be used with TRL as well with GRPO. I downloaded the grpo trainer code into TRL cookbooks, please check it out before implementing the TRL adapter and then make sure our reward functions can be used inside TRL as well. Our current TRL examples are not working yet since start and end of training is still exactly the same.

### Implementation Plan for TRL Integration

#### Overview
Based on the analysis of the TRL cookbook example and our reward-kit codebase, the goal is to create a seamless integration between reward-kit reward functions and TRL's GRPO trainer, as well as other TRL trainers.

#### Current State Analysis
1. The RewardFunction class already has a get_trl_adapter() method in reward_function.py lines 224-287
2. This adapter converts our reward functions to a format compatible with TRL (handling batch inputs)
3. TRL's GRPO trainer requires reward functions that:
   - Process batches of completions in specific formats
   - Support the specialized <think>/<answer> format from DeepSeek-R1
   - Can be combined (e.g., format_reward + accuracy_reward)
4. The existing adapter does not fully address GRPO-specific requirements

#### Implementation Tasks and Testing

1. **Enhance the TRL Adapter Implementation**
   - Update adapter to support GRPO's completion format and latest TRL version
   - Add specialized format checking for <think>/<answer> tags
   - Improve error handling and reporting for TRL compatibility

   **Testing at this stage:**
   - Unit test: Verify adapter handles different input formats (string lists, message arrays)
   - Unit test: Confirm adapter correctly processes GRPO-style outputs with think/answer tags
   - Unit test: Check that batch processing works with various reward functions
   - Edge case test: Ensure adapter gracefully handles malformed inputs, empty batches, etc.
   - Performance test: Measure overhead of the adapter vs. direct function calls

2. **Create a GRPO-Specific Format Reward**
   - Implement a specialized reward function for GRPO format compliance
   - Support customizable tag pairs (e.g., <think>/<answer>, <reasoning>/<solution>)
   - Add weighting mechanism for format vs. content rewards

   **Testing at this stage:**
   - Unit test: Verify format detection for various patterns of think/answer tags
   - Unit test: Ensure correct scoring for well-formed vs. malformed outputs
   - Unit test: Check format scoring with different tag variations and positions
   - Integration test: Combine format reward with accuracy reward to verify weighting

3. **Implement Reward Combiners for TRL**
   - Create utility for combining multiple reward functions with weights
   - Support normalized and non-normalized combining methods
   - Add GRPO-specific reward combiner that handles format+content rewards

   **Testing at this stage:**
   - Unit test: Verify weighted combinations give expected results
   - Unit test: Test normalization methods for different reward value ranges
   - Unit test: Check combiners work with both function-based and class-based rewards
   - Integration test: Verify combiners work when passed to TRL trainer

4. **Develop Dataset Preparation Utilities**
   - Create converters from HuggingFace datasets to TRL training format
   - Support custom prompt templates and formatting
   - Add system prompt insertion for GRPO-style training

   **Testing at this stage:**
   - Unit test: Verify dataset conversion preserves sample count and content
   - Unit test: Check system prompt is correctly inserted in converted datasets
   - Unit test: Ensure dataset format matches what TRL trainers expect
   - Integration test: Load converted dataset into TRL trainer to verify compatibility

5. **Create End-to-End Examples**
   - Basic example: Using a simple reward function with PPO in TRL
   - GRPO example: Full implementation matching the cookbook pattern
   - Advanced example: Custom reward function with specialized format

   **Testing at this stage:**
   - Smoke test: Verify examples run without errors
   - Minimal training test: Run 5-10 steps to ensure learning happens
   - Input/output test: Check model outputs improve according to reward function
   - Documentation test: Ensure examples work when following documented steps

6. **Add TRL Compatibility to Existing Reward Functions**
   - Update existing rewards (accuracy, length, etc.) to work seamlessly with TRL
   - Add TRL-specific parameters where needed
   - Ensure rewards can be used both standalone and with TRL

   **Testing at this stage:**
   - Unit test: Verify each reward function works with TRL adapter
   - Integration test: Test each reward with TRL in isolation
   - Combination test: Verify multiple rewards can be used together in TRL

#### Implementation Details

1. **Enhanced TRL Adapter**
```python
def get_trl_adapter(self, format_type=None, **kwargs):
    """
    Create an adapter function for use with TRL library.
    
    Args:
        format_type: Optional format to enforce ("grpo", "ppo", etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        A callable function compatible with TRL
    """
    def adapter(batch_input, batch_orig_input=None, **adapter_kwargs):
        # Detect input format (strings vs. message arrays)
        # Process according to format_type
        # Handle batched inputs efficiently
        # Return list of scores in TRL-expected format
        pass
    
    return adapter
```

2. **GRPO Format Reward**
```python
@reward_function
def grpo_format_reward(
    messages,
    original_messages=None,
    think_tag="<think>",
    answer_tag="<answer>",
    **kwargs
):
    """
    Reward function that evaluates if output follows GRPO format with think/answer tags.
    
    Args:
        messages: List of conversation messages
        original_messages: Original messages for context
        think_tag: Tag to use for reasoning section
        answer_tag: Tag to use for answer section
        
    Returns:
        EvaluateResult with score based on format compliance
    """
    # Implementation that checks for proper tag usage
    # Score based on tag presence, order, and completeness
    pass
```

3. **Reward Combiner**
```python
def combine_rewards(
    reward_functions,
    weights=None,
    normalize=True,
    trl_format=True
):
    """
    Combine multiple reward functions into a single reward function.
    
    Args:
        reward_functions: List of RewardFunction instances
        weights: Optional weights for each reward function
        normalize: Whether to normalize rewards before combining
        trl_format: Whether to return a TRL-compatible function
        
    Returns:
        Combined RewardFunction instance
    """
    # Implementation for combining reward outputs
    # Support for weighting and normalization
    pass
```

4. **Dataset Converter**
```python
def prepare_dataset_for_trl(
    dataset_name,
    split="train",
    prompt_key=None,
    response_key=None,
    system_prompt=None,
    format_template=None,
    max_samples=None
):
    """
    Prepare a HuggingFace dataset for use with TRL.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        prompt_key: Key for the prompt content
        response_key: Key for the response content
        system_prompt: Optional system prompt to prepend
        format_template: Template for formatting prompts
        max_samples: Maximum samples to include
        
    Returns:
        Dataset in TRL-compatible format
    """
    # Load and convert dataset
    # Apply formatting and templates
    # Return in TRL format
    pass
```

5. **Example of Reward-Kit with GRPO**
```python
def train_with_grpo():
    """
    Example of training with GRPO using reward-kit rewards.
    """
    # Load model and tokenizer
    # Prepare dataset
    # Define reward functions
    # Configure GRPO trainer
    # Run training
    pass
```

#### Testing Strategy
- Unit tests for each component in isolation
- Integration tests combining multiple components
- End-to-end tests with minimal training
- Regression tests against baseline implementations
- Performance benchmarks for adapter overhead

#### Deliverables
1. Enhanced TRL adapter in RewardFunction class
2. GRPO-specific reward functions and utilities
3. Dataset preparation utilities for TRL
4. Example scripts demonstrating integration
5. Comprehensive tests for all components
6. Documentation updates explaining TRL integration

## Support uploading of HuggingFace datasets to Fireworks dataset (requires Fireworks dataset API integration)

People should be able to just specify a huggingface dataset for evaluation job, and we should still make everything run
