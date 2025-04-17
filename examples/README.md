# Fireworks Reward Kit Examples

This directory contains examples demonstrating how to use the Fireworks Reward Kit.

## Examples

### Basic Rewards

- [`basic_reward.py`](./basic_reward.py) - Demonstrates how to create simple reward functions and compose them together.

### Function Calling Rewards

- [`function_calling_reward.py`](./function_calling_reward.py) - Shows how to use the built-in function calling reward to evaluate LLM function calls.

### Serving Reward Functions

- [`server_example.py`](./server_example.py) - Demonstrates how to serve a reward function as an HTTP API.

## Running the Examples

Make sure you have installed the reward-kit package in development mode:

```bash
pip install -e .
```

Then run any example with:

```bash
python examples/basic_reward.py
```

### Testing the Server Example

After starting the server with:

```bash
python examples/server_example.py
```

You can test it with a curl command:

```bash
curl -X POST http://localhost:8000/reward \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me about RLHF"},
      {"role": "assistant", "content": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align language models with human preferences. It involves training a reward model using human feedback and then fine-tuning an LLM using reinforcement learning to maximize this learned reward function."}
    ]
  }'
```

## Next Steps

After exploring these examples, you can:

1. Create your own reward functions tailored to your specific use case
2. Deploy them using the Fireworks platform
3. Use them in RL fine-tuning jobs

Check the main README file for more detailed information on the API and deployment options.