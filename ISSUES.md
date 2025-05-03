# Goal
- clean up example for evaluation_preview_example.py to read from file
- improve deployment interface
- setup CLI

Please make sure you run the example. And make sure the preview example actually runs before you do anything, and after you declare victory

```
source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/evaluation_preview_example.py
```

## Preview API is hard to use, move to cli API for example instead
Current to create a API preview, it is really weird to use because we have all the file content as string in the file. It does run

```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ source .venv/bin/activate && FIREWORKS_API_KEY=$DEV_FIREWORKS_API_KEY FIREWORKS_API_BASE=https://dev.api.fireworks.ai python examples/evaluation_preview_example.py
Previewing evaluation...
INFO:reward_kit.evaluation:Loaded 1 Python files for metric 'word_count' from /home/bchen/home/reward-kit/tmp_metric
INFO:reward_kit.evaluation:Loaded 2 samples from ./samples.jsonl
INFO:reward_kit.evaluation:Previewing evaluator using API endpoint: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators:previewEvaluator with account: pyroworks-dev
Evaluation Preview Results
------------------------
Total Samples: 2
Total Runtime: 10222 ms

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
INFO:reward_kit.evaluation:Loaded 1 Python files for metric 'word_count' from /home/bchen/home/reward-kit/tmp_metric
INFO:reward_kit.evaluation:Creating evaluator 'word-count-eval' for account 'pyroworks-dev'...
INFO:reward_kit.evaluation:Evaluator 'word-count-eval' already exists, deleting and recreating...
INFO:reward_kit.evaluation:Successfully deleted evaluator 'word-count-eval'
INFO:reward_kit.evaluation:Successfully created evaluator 'word-count-eval'
Created evaluator: accounts/pyroworks-dev/evaluators/word-count-eval
```

but I think it is not that clean, make sure to
- move all the content of the preview files into a separate folder
- then just try to read the content of the file and try to deploy

## Deployment example is very unfriendly

for examples/deploy_example.py, right now the authentication complications are not handled properly in the SDK and handled in the example instead.
we should instead properly refactor the authentication logic into the SDK, and then add all the relevant unittests to cover for different situation for auth
also we should not hard code the pyroworks-dev situation, that was just an one off.

check reward_kit/auth.py, I am half way through, if it is not useful feel free to remove any code you seem unhelpful, and get me to a place where deployment is super simple

## Setup both preview and deployment to be CLI based
Ideally I can just run a cli to preview and deploy examples. So please
- setup the code so I can just run a CLI command to
  - preview based on code and samples
  - deploy an evaluator based on code
