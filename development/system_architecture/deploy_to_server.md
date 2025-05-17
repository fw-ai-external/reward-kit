# Creating a server side reward function

Currently for both reviewing local and deploying to Fireworks, we need to move the reward function to the server side.

## Example: Deploying a Reward Function to Fireworks as an Evaluation

```python
from typing import List, Dict, Optional
from reward_kit import reward_function
from reward_kit.models import Message, EvaluateResult

@reward_function
def helpfulness_reward(
    messages: List[Message],
    **kwargs
) -> EvaluateResult:
    """
    Evaluates the helpfulness of an assistant response based on
    length and keyword presence.
    """
    # Get the assistant's response
    if not messages or messages[-1].role != "assistant":
        return EvaluateResult(
            score=0.0,
            metrics={
                "error": {
                    "score": 0.0,
                    "reason": "No assistant response found"
                }
            }
        )

    response = messages[-1].content
    metrics = {}

    # 1. Length check - reward longer responses up to a point
    length = len(response)
    length_score = min(length / 500.0, 1.0)  # Cap at 500 chars
    metrics["length"] = {
        "score": length_score * 0.3,  # 30% weight
        "reason": f"Response length: {length} chars"
    }

    # 2. Keyword analysis for helpfulness indicators
    helpful_keywords = ["specifically", "example", "for instance", "steps", "how to"]
    keyword_count = sum(1 for kw in helpful_keywords if kw.lower() in response.lower())
    keyword_score = min(keyword_count / 3.0, 1.0)  # Cap at 3 keywords
    metrics["keywords"] = {
        "score": keyword_score * 0.7,  # 70% weight
        "reason": f"Found {keyword_count} helpful keywords"
    }

    # Calculate final score as weighted sum of metrics
    final_score = sum(metric["score"] for metric in metrics.values())

    return EvaluateResult(score=final_score, metrics=metrics)

# Deploy the reward function to Fireworks
if __name__ == "__main__":
    # Option 1: Basic deployment with default parameters
    evaluation_id = helpfulness_reward.deploy(
        name="helpfulness-v1",
        description="Evaluates response helpfulness based on length and keywords"
    )
    print(f"Deployed evaluation with ID: {evaluation_id}")

    # Option 2: Advanced deployment with custom provider
    evaluation_id_advanced = helpfulness_reward.deploy(
        name="helpfulness-v1-claude",
        description="Helpfulness evaluation using Claude model",
        providers=[
            {
                "providerType": "anthropic",
                "modelId": "claude-3-sonnet-20240229"
            }
        ]
    )
    print(f"Deployed evaluation with custom provider: {evaluation_id_advanced}")

    # The evaluation_id can now be used in Fireworks RL training jobs
    print("Use this in your RL training job:")
    print(f"firectl create rl-job --reward-endpoint \"https://api.fireworks.ai/v1/evaluations/{evaluation_id}\"")
```

This example demonstrates how to define a reward function that evaluates response helpfulness and deploys it to Fireworks as an evaluation using the `@reward_function` decorator and `.deploy()` method.

## Relevant APIs

Here are part of the fireworks.swagger.yaml file that is relevant to our case. At the top level, we are creating an evaluation with the following URL:

```
  /v1/accounts/{account_id}/evaluations:
    get:
      summary: List Evaluations
      operationId: Gateway_ListEvaluations
      responses:
        '200':
          description: A successful response.
          schema:
            $ref: '#/definitions/gatewayListEvaluationsResponse'
      parameters:
      - name: pageSize
        in: query
        required: false
        type: integer
        format: int32
      - name: pageToken
        in: query
        required: false
        type: string
      - name: filter
        in: query
        required: false
        type: string
      - name: orderBy
        in: query
        required: false
        type: string
      - name: account_id
        in: path
        required: true
        type: string
        description: The Account Id
      tags:
      - Gateway
    post:
      summary: Create Evaluation
      operationId: Gateway_CreateEvaluation
      responses:
        '200':
          description: A successful response.
          schema:
            $ref: '#/definitions/gatewayEvaluation'
      parameters:
      - name: body
        in: body
        required: true
        schema:
          $ref: '#/definitions/GatewayCreateEvaluationBody'
      - name: account_id
        in: path
        required: true
        type: string
        description: The Account Id
      tags:
      - Gateway
```

the body is defined here:
```
  GatewayCreateEvaluationBody:
    type: object
    properties:
      evaluation:
        $ref: '#/definitions/gatewayEvaluation'
      evaluationId:
        type: string
    required:
    - evaluation
```

```yaml
  gatewayEvaluation:
    type: object
    properties:
      name:
        type: string
        title: Current fields in your proto
        readOnly: true
      createTime:
        type: string
        format: date-time
        readOnly: true
      createdBy:
        type: string
        readOnly: true
      status:
        $ref: '#/definitions/gatewayStatus'
        readOnly: true
      evaluationType:
        type: string
        title: string llm_evaluator_prompt = 6;
      description:
        type: string
        title: Optional description of the evaluation
      providers:
        type: array
        items:
          type: object
          $ref: '#/definitions/gatewayProvider'
        title: One or more providers to use
      assertions:
        type: array
        items:
          type: object
          $ref: '#/definitions/gatewayAssertion'
        title: One or more assertions to evaluate
      updateTime:
        type: string
        format: date-time
        description: The update time for the evaluation.
        readOnly: true
    title: 'Next ID: 11'
    required:
    - evaluationType
    - providers
    - assertions
```

for assertions, we need to create the code assertion.

```
  gatewayAssertion:
    type: object
    properties:
      assertionType:
        $ref: '#/definitions/AssertionAssertionType'
      llmAssertion:
        $ref: '#/definitions/gatewayLLMAssertion'
      codeAssertion:
        $ref: '#/definitions/gatewayCodeAssertion'
      metricName:
        type: string
    title: 'We are doing auto generated GORM with JSON serializer and oneof doesn''t
      work

      so I am doing enums + just flat fields'
    required:
    - assertionType
```

for code assertion, we need to create a python assertion.

```
  gatewayCodeAssertion:
    type: object
    properties:
      language:
        type: string
        title: Language of the code (python/javascript)
      code:
        type: string
        title: The code to execute
      expectedOutput:
        type: string
        title: Optional expected output
      options:
        $ref: '#/definitions/CodeAssertionExecutionOptions'
    required:
    - language
    - code
```
