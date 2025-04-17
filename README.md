# Introducing the Fireworks Reward Kit: Simpler, Composable Reward Modeling for LLM RL Fine-Tuning
Reinforcement Learning from AI Feedback (RLAIF), is a powerful technique for aligning Large Language Models (LLMs) to desired behaviors. However, implementing the reward modeling component can often be complex, tightly coupled to specific RL frameworks, and difficult to scale or iterate upon.

Today, we're excited to introduce the **Fireworks Reward Kit**, a Python library designed to drastically simplify how you define, test, deploy, and *use* reward functions for LLM fine-tuning, including launching full RL jobs directly on the Fireworks platform.

**Our core philosophy:** Your reward logic should be:

- **Python-Native:** Defined as straightforward Python functions.
- **Context-Aware:** Easily access conversational history.
- **Insightful:** Return not just a final score, but components for analysis.
- **Composable:** Build complex logic from modular parts.
- **Testable:** Easy to test locally using standard Python tools.
- **Flexible Deployment:** Run locally, self-host via a simple command, leverage powerful Fireworks-hosted reward models, or deploy seamlessly to Fireworks managed infrastructure.
- **Integrated:** Directly usable within Fireworks RL training jobs.

### The Problem: Complexity in RL Reward Signals

Traditional RL setups often require reward functions to fit specific batch-oriented signatures, making integration complex. Incorporating external models, databases, vector stores, or complex heuristics adds further hurdles. Scaling these computations and orchestrating RL jobs requires significant infrastructure effort. Debugging reward signals ("reward hacking") is difficult without visibility into scoring components.

### The Solution: Simple, Composable, Deployable Python Functions with Rich Outputs

The Fireworks RL Reward SDK lets you define your reward logic using a natural, context-aware, **pointwise** Python function signature that returns both a final score and its components:

```python
# In the reward_kit library
from typing import List, Dict, Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class MetricRewardOutput:
    score: float
    reason: Optional[str]

@dataclass_json
@dataclass
class RewardOutput:
    score: float
    metrics: Dict[str, MetricRewardOutput]
```

and then for the reward function you need to define
```python
from reward_kit import RewardOutput
def calculate_base_score(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """Calculates a basic score based on length and keywords."""
    last_response = messages[-1]['content']
    metrics = {}
    
    # Evaluate helpfulness
    is_helpful = "helpful" in last_response.lower()
    metrics["helpfulness"] = MetricRewardOutput(
        score=0.5 if is_helpful else 0.0,
        reason="Contains helpful keyword" if is_helpful else "Missing helpful keyword"
    )
    
    # Evaluate length
    is_long_enough = len(last_response) > 50
    metrics["length_bonus"] = MetricRewardOutput(
        score=0.5 if is_long_enough else 0.0,
        reason="Response length sufficient" if is_long_enough else "Response too short"
    )
    
    # ... potentially more complex logic ...
    final_score = sum(metric.score for metric in metrics.values())
    return RewardOutput(score=final_score, metrics=metrics)
```

This signature returns a `(float, Dict[str, float])` tuple, enabling fine-grained analysis against reward hacking.

### Building Composable Rewards

Combine multiple reward functions for more sophisticated evaluations.

```python
from reward_kit import RewardFunction

def calculate_safety_score(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """Calculates a safety score (example: penalizes forbidden words)."""
    last_response = messages[-1]['content'].lower()
    components = {}
    penalty = 0.0
    if "unsafe_word" in last_response:
        penalty = -1.0
    components["safety_penalty"] = penalty
    # A safety score might range from 0 (unsafe) to 1 (safe),
    # here we just return the penalty component directly.
    final_score = penalty # In this simple case, final score is just the penalty
    return final_score, components

# Now, compose them:
def combined_reward(
    messages: List[Dict[str, str]],
    original_messages: List[Dict[str, str]],
    **kwargs
) -> RewardOutput:
    """Combines base score and safety score."""
    final_base, components_base = calculate_base_score(messages, original_messages, **kwargs)
    final_safety, components_safety = calculate_safety_score(messages, original_messages, **kwargs)

    # Combine scores and components
    all_components = {**components_base, **components_safety}
    final_score = final_base + final_safety # Adjust aggregation as needed

    return final_score, all_components
```

This approach keeps individual reward functions focused and allows combining them easily. *Note: While this example uses local composition, one of the composed functions (e.g., `calculate_safety_score`) could itself be an instance of `RewardFunction(mode="remote", endpoint="...")` if you had deployed it separately.*

### Out-of-the-Box Rewards: Function Calling Example

Accelerate development with pre-built functions for common tasks like evaluating LLM-generated function calls.

```python
from reward_kit import RewardFunction
from reward_kit.rewards import function_calling

expected_call = { # Define expected structure (as before)
    "name": "get_weather",
    "arguments": {"location": {"type": "string"}, "unit": {"type": "string"}}
}

# Instantiate the OOTB reward function
fn_call_reward_model = RewardFunction(
    func=function_calling.match_function_call,
    mode="local", # Runs locally using the SDK's built-in logic
    expected_call_schema=expected_call,
    argument_match_strictness="exact"
)

def my_agent_reward(messages, original_messages, **kwargs) -> RewardOutput:
    is_function_call, parsed_call = parse_potential_function_call(messages[-1])
    if is_function_call:
        # Returns (score, components) based on match quality
        return fn_call_reward_model(
            messages=messages, original_messages=original_messages,
            parsed_arguments=parsed_call.get('arguments',{}),
            function_name=parsed_call.get('name')
        )
    else:
        return calculate_base_score(messages, original_messages) # Fallback

# Wrap 'my_agent_reward'
# agent_reward_wrapper = RewardFunction(func_path="my_rewards.py::my_agent_reward", mode="local")

```

### Using Fireworks Hosted Reward Models

Leverage powerful, pre-trained reward models hosted directly on the Fireworks platform without needing to manage them yourself. Example using a hypothetical Nemotron-340B reward model:

```python
from reward_kit import RewardFunction

# Instantiate by pointing to a Fireworks-hosted model ID
# (Ensure your Fireworks API key is configured)
nemotron_reward_model = RewardFunction(
    # No func_path needed, using a known Fireworks model identifier
    model_id="fireworks/nemotron-340b-reward",
    mode="fireworks_hosted" # Special mode indicating a Fireworks managed model
)

# Use it like any other reward function
test_orig = [{"role": "user", "content": "Write a poem about stars."}]
test_msgs = test_orig + [{"role": "assistant", "content": "..." # LLM Response
                        }]

try:
    # This call goes to the Fireworks API endpoint for the Nemotron reward model
    final_score, components = nemotron_reward_model(
        messages=test_msgs,
        original_messages=test_orig
        # Specific models might accept/require additional args via kwargs
    )
    print(f"Nemotron Reward Score: {final_score}")
    # Components dict structure depends on the specific hosted model
    print(f"Nemotron Components: {components}")

except Exception as e:
    print(f"Error calling hosted reward model: {e}")

```

This allows you to easily incorporate state-of-the-art reward signals into your workflow, scaled and managed by Fireworks.

### Get Value Immediately: Local Development & Testing

Use `mode="local"` for instant feedback on your custom functions:

```python
# Instantiate your custom combined reward
reward_model_local = RewardFunction(func_path="my_rewards.py::combined_reward", mode="local")

test_orig = [{"role": "user", "content": "Explain RLHF safely."}]
test_msgs = test_orig + [{"role": "assistant", "content": "RLHF is helpful..."}]

final_score, components = reward_model_local(messages=test_msgs, original_messages=test_orig)
print(f"Calculated local reward: {final_score}, Components: {components}")

```

### Bridging to RL Libraries like TRL

```python
trl_compatible_reward_fn = reward_model_local.get_trl_adapter()
# Pass to PPOTrainer: reward_fn=trl_compatible_reward_fn
```

### Deployment Options: Scale Your Custom Rewards

To make deploying and scaling your custom reward functions seamless, we provide a simple decorator-based approach, inspired by familiar patterns in frameworks like Ray Serve and Modal. This allows you to focus on your reward logic in standard Python while easily transitioning to a scalable deployment.

**1. Define Your Reward Function with `@reward_function`**

Simply write your reward logic as a standard Python function and apply the `@reward_function` decorator provided by our library. The decorator handles the necessary preparations for deployment without altering your core logic's signature or behavior for local execution.

```python
from reward_kit import reward_function

@reward_function
def combined_reward(messages: Dict[str, Any], *args, **kwargs) -> float:
    """
    Calculates a reward based on completion length and keyword presence.
    (This is just an example, your logic can be arbitrarily complex)
    """
    length_score = min(len(messages[-1].content) / 100.0, 1.0) # Score based on length (up to 100 chars)
    keyword_present = "important" in completion.lower()
    
    final_score = 0.7 * length_score + 0.3 * (1.0 if keyword_present else 0.0)
    
    # You can access anything passed in metadata if needed
    if metadata.get("user_preference") == "concise":
        final_score *= 0.9 # Penalize verbosity slightly if user prefers concise

    return final_score
```

**2. Local Execution and Testing**

Crucially, your decorated function still works exactly like a regular Python function. You can call it directly to test your logic locally before considering deployment.

```python

messages = [{"user": "Explain the concept of reinforcement learning.", "assistant": "RL involves agents learning through trial and error via rewards. It's important."
metadata_example = {"user_preference": "concise"}

local_score = combined_reward(messages, metadata_example)
print(f"Calculated local reward score: {local_score}")
# Output: Calculated local reward score: 0.568... (example value)
```

**3. Deploying for Scalability**

When you're ready to scale, the `@reward_function` decorator automatically adds a `.deploy()` method to your function object. Calling this method initiates the deployment process on the Fireworks platform.

You can pass configuration options directly to `.deploy()` to control resources, scaling behaviour, environment variables, secrets, and more.

```python
# --- Deployment Example ---
deployment_config = {
    "name": "combined-reward-prod",
}
print("Submitting reward function for deployment...")
deployment_handle = combined_reward.deploy(**deployment_config)
print(f"Deployment submitted. Handle/ID: {deployment_handle}")

```

The `.deploy()` method takes care of:

- Packaging your function code and its dependencies.
- Provisioning the specified cloud resources.
- Setting up autoscaling based on your configuration.
- Making your reward function available as a secure, scalable endpoint within the Fireworks ecosystem (e.g., for use in RLHF pipelines or model evaluation).

This decorator approach simplifies the transition from local development to scalable deployment, letting you manage your reward logic and its operationalization together cleanly.

### Using Your Deployed/Hosted Reward Function

Interact with remote functions (either Fireworks-deployed, self-hosted, or Fireworks-hosted models):

```python
# Example: Using the function deployed to Fireworks
remote_reward_model = RewardFunction(name=evaluation_name, mode="remote")
final_score, components = remote_reward_model(messages=test_msgs, original_messages=test_orig)
print(f"Calculated remote reward: {final_score}, Components: {components}")

# Example: Using the self-hosted function (replace with your actual URL)
self_hosted_reward_model = RewardFunction(endpoint="<http://127.0.0.1:8000/reward>", mode="remote")
# Note: Authentication might be needed depending on your self-hosted setup
final_score_sh, components_sh = self_hosted_reward_model(messages=test_msgs, original_messages=test_orig)
print(f"Calculated self-hosted reward: {final_score_sh}, Components: {components_sh}")

# Example: Using the Fireworks-hosted Nemotron model (from earlier)
# nemotron_reward_model = RewardFunction(model_id="fireworks/nemotron-340b-reward", mode="fireworks_hosted")
# score_nem, comps_nem = nemotron_reward_model(...)

# Get TRL adapter for any remote/hosted model
remote_trl_adapter = remote_reward_model.get_trl_adapter()
# Use with TRL: ppo_trainer_remote = PPOTrainer(..., reward_fn=remote_trl_adapter)

```

### Launching Fireworks RL Jobs with Your Reward

Use *any* accessible reward endpoint (Fireworks-deployed, publicly accessible self-hosted, or even certain Fireworks-hosted model endpoints if supported) in a Fireworks RL job:

```bash
# Example using firectl with a Fireworks-deployed endpoint
firectl create rl-job \\
    --model <your_base_model_name_or_id> \\
    --training-file <path_or_uri_to_your_training_data> \\
    # Use the URL from .deploy() or your known endpoint
    --reward-endpoint "<paste_your_endpoint_url_here>" \\
    --ppo-config <path_to_ppo_config.yaml> \\
    --output-model-name <your_finetuned_model_name> \\
    # ... other RL parameters ...

```

The RL service calls the specified endpoint to get rewards during training.

### End-to-End RL Tuning Workflow

1. **Define:** Write Python logic `(messages, original_messages) -> (float, Dict[str, float])`.
2. **Compose (Optional):** Combine modular reward functions.
3. **Leverage (Optional):** Instantiate `RewardFunction` with `mode="fireworks_hosted"` to use powerful models like Nemotron.
4. **Test Locally:** Use `mode="local"` for rapid iteration. Check score and components.
5. **Adapt (Optional):** Use `.get_trl_adapter()` for TRL/batch library compatibility.
6. **Deploy/Serve:**
    - **Fireworks Managed:** Use `.deploy()` for a scalable endpoint. Note the URL.
    - **Self-Host:** Use `fireworks-reward serve ...` command. Note the URL and manage the server.
    - <offline discussion>
        - this needs more detail
        - reverse tunnel or cloud run
            - for reverse tunnel
                - Fireworks do the reverse tunnel
                    - Open a connection with a intermediary
                - ngrok do the reverse tunnel
                    - we authenticate with API Key
                    - when you spin up the fastapi server, create an FIREWORKS API key you can share with FW
                        - we can do the create secret in the background and send it to FW
                    - For now let’s do ngrok
    - **Use Fireworks Hosted:** Use the `RewardFunction(mode="fireworks_hosted", ...)` directly.
7. **Launch RL Job:** Use the appropriate reward endpoint URL with Fireworks RL
    1. `firectl create evaluation --reward-endpoint`
        1. get evaluation id
    2.  (`firectl create rl-job --evaluation-id ...`).
8. **Monitor & Iterate:** Track RL job progress. Use **Evaluator Logs** (for Fireworks-deployed functions) or your own logging (for self-hosted) to debug. Analyze components to refine logic.

### Why This Matters for MLEs

- **Simplicity & Familiarity:** Standard Python functions, less boilerplate.
- **Rich Reward Signals:** Component dictionary aids debugging and analysis.
- **Composability:** Build sophisticated logic modularly.
- **Ultimate Flexibility:**
    - Run locally for testing.
    - Use powerful **Fireworks-hosted** reward models (e.g., Nemotron) out-of-the-box.
    - Deploy custom logic to **Fireworks managed infra** easily.
    - **Self-host** custom logic using the SDK's server command for full control. Reduces vendor lock-in.
- **Managed Infrastructure Benefits:** Secrets, basic dependencies, scaling handled by Fireworks deployment.
- **Seamless Scaling:** One command deployment or leverage pre-scaled hosted models.
- **End-to-End Integration:** Direct use of reward endpoints in managed Fireworks RL jobs.
- **Observability:** Dedicated logs for Fireworks-deployed functions.
- **TRL Compatibility:** Adapters bridge the gap to popular libraries.
- **Cost-Effective:** Managed deployment currently free; competitive pricing for hosted models.

### Get Started Today!

The Fireworks RL Reward SDK offers a pragmatic, flexible, and powerful approach to RL reward modeling and execution, letting you choose the right level of control and convenience.

- **Install the SDK:** `pip install reward_kit` (*Use actual name*)
- **Check out the Docs:** [Link to your documentation] (Including details on deployment options, `fireworks-reward serve`, hosted models, secrets, TRL adapter)
- **Try the Examples:** [Link to examples repository/directory]

We're excited to see how you use it to align your models! Share your feedback and happy tuning!
