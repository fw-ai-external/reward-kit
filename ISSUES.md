# Goal
Replica the evaluation preview and deploy logic in firectl in the SDK

## Current issue

I am not using the new API properly, here is the latest error running things end to end

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
Using auth token from environment
Token starts with: fw_3Zeaces...
Using account ID from settings: pyroworks-dev
INFO:reward_kit.reward_function:Making request to: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators (using API base: https://dev.api.fireworks.ai)
INFO:reward_kit.reward_function:Using account_id: pyroworks-dev
INFO:reward_kit.reward_function:Auth token present: True
INFO:reward_kit.reward_function:Deploying reward function 'informativeness_reward' as evaluation 'informativeness-v1'...
ERROR:reward_kit.reward_function:Error deploying evaluation: 400 Client Error: Bad Request for url: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators
ERROR:reward_kit.reward_function:Response: {
  "code": 3,
  "details": [],
  "message": "proto: (line 1:2): unknown field \"evaluationId\""
}
Traceback (most recent call last):
  File "/home/bchen/home/reward-kit/examples/deploy_example.py", line 194, in <module>
    deploy_to_fireworks()
  File "/home/bchen/home/reward-kit/examples/deploy_example.py", line 158, in deploy_to_fireworks
    evaluation_id = informativeness_reward.deploy(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bchen/home/reward-kit/reward_kit/reward_function.py", line 486, in deploy
    response.raise_for_status()
  File "/home/bchen/home/reward-kit/.venv/lib/python3.12/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators
```

## REST-ful api reference
- check the super large fireworks.swagger.yaml, you can start with   /v1/accounts/{account_id}/evaluators:

## Evaluation preview

So if you want to preview the code (needs main.py that implements evaluate()), you can build the latest firectl-admin and do this

```
firectl-admin preview eval --metric-folder <metric_name>=/path/to/python-folder --sample-file /path/to/file.jsonl
```

## Evaluation creation
after preview, you can then create with higher confidence, let me grab the command
```
firectl-admin create eval --metric-folder metric=/path/to/folder/ <eval_id>
```
Or if it's one folder generating multiple metrics, taht's called --multi-metrics
```
firectl-admin create eval --multi-metrics --folder /path/to/folder/ <eval_id>
```

## How to query existing endpoint for example evaluator
Note that this example currently only works on dev.api.fireworks.ai, we probably need an env variable to point to dev and prod (prod is api.fireworks.ai)

```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ cd /home/bchen/home/reward-kit && source .venv/bin/activate && curl -s "https://dev.api.fireworks.ai/v1/accounts/pyroworks-dev/evaluators" -H "Authorization: Bearer $DEV_FIREWORKS_API_KEY"                                                              
{
  "evaluators": [
    {
      "createTime": "2025-04-25T18:41:06.919535Z",
      "createdBy": "bryanlin@fireworks.ai",
      "criteria": [
        {
          "codeSnippets": {
            "fileContents": {
              "helper.py": "import random\n\n\ndef random1() -\u003e bool:\n    return random.random() \u003e 0.5\n",
              "main.py": "from .helper import random1\n\n\ndef evaluate(entry: dict) -\u003e dict:\n    messages = entry[\"messages\"]\n    assistant_response = next((item for item in reversed(messages) if item.get(\"role\") == \"assistant\"), None)\n    success = \"help\" in assistant_response.get(\"content\")\n    if success and random1():\n        return {\n            \"score\": 1,\n            \"reason\": \"Looks good to me\",\n        }\n    else:\n        return {\n            \"score\": 0,\n            \"reason\": (\n                \"assistant was not helpful\"\n                if \"help\" not in assistant_response.get(\"content\")\n                else \"API response not successful\"\n            ),\n        }\n"
            },
            "language": "python"
          },
          "description": "",
          "name": "helpful",
          "type": "CODE_SNIPPETS"
        }
      ],
      "description": "",
      "displayName": "",
      "multiMetrics": false,
      "name": "accounts/pyroworks-dev/evaluators/boyan-evaluator-multi-criteria-1",
      "requirements": "",
      "rollupSettings": null,
      "state": "STATE_UNSPECIFIED",
      "updateTime": "2025-04-25T18:41:06.919535Z"
    },
    {
      "createTime": "2025-04-25T18:15:46.596750Z",
      "createdBy": "bryanlin@fireworks.ai",
      "criteria": [
        {
          "codeSnippets": {
            "fileContents": {
              "main.py": "from utils import random1, random2\n\n\ndef evaluate(entry: dict) -\u003e dict:\n    messages = entry[\"messages\"]\n    assistant_response = next((item for item in reversed(messages) if item.get(\"role\") == \"assistant\"), None)\n    success = \"help\" in assistant_response.get(\"content\")\n    if success:\n        helpful = {\n            \"success\": True,\n            \"score\": 1,\n            \"reason\": \"Looks good to me\",\n        }\n    else:\n        helpful = {\n            \"success\": False,\n            \"score\": 0,\n            \"reason\": (\n                \"assistant was not helpful\"\n                if \"help\" not in assistant_response.get(\"content\")\n                else \"API response not successful\"\n            ),\n        }\n\n    if random1():\n        r1 = {\n            \"score\": 1,\n            \"reason\": \"random1 returned True\",\n        }\n    else:\n        r1 = {\n            \"score\": 0,\n            \"reason\": \"random1 returned False\",\n        }\n\n    if random2():\n        r2 = {\n            \"score\": 1,\n            \"reason\": \"random2 returned True\",\n        }\n    else:\n        r2 = {\n            \"score\": 0,\n            \"reason\": \"random2 returned False\",\n        }\n\n    return {\n        \"helpful\": helpful,\n        \"random1\": r1,\n        \"random2\": r2,\n    }\n",
              "utils.py": "import random\n\n\ndef random1() -\u003e bool:\n    return random.random() \u003e 0.5\n\n\ndef random2() -\u003e bool:\n    return random.random() \u003e 0.5\n"
            },
            "language": "python"
          },
          "description": "",
          "name": "eval",
          "type": "CODE_SNIPPETS"
        }
      ],
      "description": "",
      "displayName": "",
      "multiMetrics": true,
      "name": "accounts/pyroworks-dev/evaluators/boyan-evaluator-multi-metrics-1",
      "requirements": "",
      "rollupSettings": null,
      "state": "STATE_UNSPECIFIED",
      "updateTime": "2025-04-25T18:15:46.596750Z"
    }
  ],
  "nextPageToken": "",
  "totalSize": 2
```

## Where existing evaluator is wrong

Current interface assumes

```
def evaluate(entry: dict)
```

we think it is too complicated and will move to

```
def evaluate(messages, original_messages, tools, **kwargs)
```
