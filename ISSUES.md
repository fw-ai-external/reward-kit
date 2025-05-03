# Goals
- deploy is not working even though it has all the right permissions - FIXED

## Deploy is now working - FIXED

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