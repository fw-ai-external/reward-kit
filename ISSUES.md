# Goal
Make the evaluator authoring experience better
- better typing for
- better deployment and preview flow

## Better typging
Right now an evaluator looks like this:
```
import numpy as np
import pydantic

def evaluate(messages: list[dict], **kwargs) -> dict:
    """
    Evaluates a single entry from the dataset. This function is required in the `main.py` file.
    If you turn on `multi_metrics`, this function should report dict from metric_name to metric_result.
    
    Args:
        messages: A list of dictionaries representing a single line from the dataset jsonl file.
        kwargs: Additional keyword arguments. Highly recommended to not remove this due to potential more keywords being passed to the function.
    Returns:
        dict: Evaluate result that should include success, score, and reason
        If `multi_metrics` is on, the result should be a dict from metric_name to metric_result.
    """
    return {
        "metric_name_1": {
            "success": True,
            "score": 1.0,
            "reason": "This is the eval result for metric_name_1"
        },
        "metric_name_2": {
            "success": False,
            "score": 0.0,
            "reason": "This is the eval result for metric_name_2"
        }
    }
```

I want to have some types so that input and output doesn't change, but the authoring experience is more typed. Here is my discussion with o3 earlier

<discussion>
Sure—this is exactly the “keep-old-API / strict-inside” pattern.  
Below is one way to wrap **any** evaluator so that:

* **Call-sites stay untouched** → they still pass `list[dict]` and get a plain `dict`.
* **Authors write with models** → inside, everything is real-typed Pydantic.

---

### 1.  Define the canonical models

```python
# models.py
from typing import Dict, List, Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class MetricResult(BaseModel):
    success: bool
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str

class EvaluateResult(BaseModel):
    """
    The whole return value:  metric_name -> MetricResult
    """
    __root__: Dict[str, MetricResult]
```

---

### 2.  The decorator that *coerces in* and *dumps out*

```python
# typed_interface.py
from functools import wraps
from typing import Any, Callable, Dict, List

from pydantic import TypeAdapter, ValidationError

from .models import Message, EvaluateResult

_msg_adapter = TypeAdapter(List[Message])
_res_adapter = TypeAdapter(EvaluateResult)

def typed_interface(
    func: Callable[[List[Message], ...], EvaluateResult]
) -> Callable[[List[dict], ...], Dict[str, Dict[str, Any]]]:
    """
    Wrap an `evaluate`-style function so callers still use raw JSON-ish types.
    """
    @wraps(func)
    def wrapper(messages: List[dict], **kwargs):
        # 1.  Validate / coerce the incoming list[dict]  →  list[Message]
        try:
            typed_messages = _msg_adapter.validate_python(messages)
        except ValidationError as err:
            raise ValueError(f"Input messages failed validation:\n{err}") from None

        # 2.  Call the author's function
        result = func(typed_messages, **kwargs)

        # Author might return EvaluateResult *or* a bare dict → coerce either way
        try:
            result_model = _res_adapter.validate_python(result)
        except ValidationError as err:
            raise ValueError(f"Return value failed validation:\n{err}") from None

        # 3.  Dump back to a plain dict for the outside world
        return _res_adapter.dump_python(result_model, mode="json")

    return wrapper
```

---

### 3.  Authoring an evaluator

```python
# main.py
from typing import List
from typed_interface import typed_interface
from models import Message, EvaluateResult, MetricResult

@typed_interface
def evaluate(messages: List[Message], **kwargs) -> EvaluateResult:
    # ...your real logic...
    return EvaluateResult({
        "metric_name_1": MetricResult(success=True,  score=1.0, reason="Passed"),
        "metric_name_2": MetricResult(success=False, score=0.0, reason="Failed"),
    })
```

---

### 4.  What a caller sees

```python
raw_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]

result = evaluate(raw_messages)
print(result)
# -> {
#      'metric_name_1': {'success': True,  'score': 1.0, 'reason': 'Passed'},
#      'metric_name_2': {'success': False, 'score': 0.0, 'reason': 'Failed'}
#    }
```

No Pydantic objects leak across the boundary.

---

### Why this works well

| Aspect | What happens |
|--------|--------------|
| **Input flexibility** | A caller can still serialise to JSON, send over a network, etc. |
| **Dev-experience**    | Inside `evaluate`, you get full autocomplete, validation, and intellisense. |
| **Zero boilerplate for authors** | They add one decorator and use typed models. |
| **Safety net**        | If either the input or output deviates from the contract, the wrapper raises a clear `ValueError` with Pydantic’s detailed trace—no silent failures. |

---

#### Optional niceties

* **Custom error class** – raise your own `ProtocolValidationError` instead of `ValueError`.
* **Speed** – `TypeAdapter` avoids model construction for every item; good for large datasets.
* **Strict mode toggles** – expose a `strict: bool = False` argument in `typed_interface` to allow looser coercion during early prototyping.
* **Automatic JSON Schema** – `EvaluateResult.model_json_schema()` can drive docs for contributors.

With this pattern you get the best of both worlds: a rock-solid, typed interior and a perfectly backward-compatible exterior.
</discussion>

For this part, we will need to install the decorator we create here on the remote server, so test everything locally first. We already have decorator
```
from reward_kit import reward_function, RewardOutput, MetricRewardOutput
```
so please don't create new types, just modify existing types to make this happen.

## Better deployment experience
### Interface
Deployment right now is based on a string, you can see that in examples/evaluation_preview_example.py. Our deploy interface is cleaner right now, check examples/deploy_example.py, but yeah please help me make the preview interface cleaner.

### Multi file evaluators
Right now the API technically already supports multi file, but then we don't really leverage that, ideally for both preview and deploy, we point a command to a folder, and then it runs and we can preview/deploy it. Can you help me with that?