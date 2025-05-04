# Issues and Tasks

## ✅ Outdated examples (FIXED)
- examples/basic_reward.py
  - should be just messages
- system_architecture/deploy_to_server.md

## ✅ Not having a single script to run all the examples as if they are end to end tests (FIXED)

Added `examples/run_all_examples.py` which can run all examples in different configurations:

```
# Run all examples
python examples/run_all_examples.py

# Skip deployment examples
python examples/run_all_examples.py --skip-deploy

# Skip E2B examples
python examples/run_all_examples.py --skip-e2b

# Run a specific example
python examples/run_all_examples.py --only=basic_reward.py

# List all available examples
python examples/run_all_examples.py --list
```

Previous command for manual testing (reference only):
```
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/deploy_example.py 
```