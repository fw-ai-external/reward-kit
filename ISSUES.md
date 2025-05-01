# Goal
- makes the type more correct

and make sure we use local venv to do this in case you need to install any dependencies

## Input is untyped, reward function can be both typed and untyped

I am looking at examples/typed_reward_function_example.py and I think this example is wrong and doesn't even run right now

```
(.venv) (base) bchen@dev-modeling:~/home/reward-kit(main)$ source .venv/bin/activate && python examples/typed_reward_function_example.py
Traceback (most recent call last):
  File "/home/bchen/home/reward-kit/examples/typed_reward_function_example.py", line 141, in <module>
    test_reward_function()
  File "/home/bchen/home/reward-kit/examples/typed_reward_function_example.py", line 125, in test_reward_function
    result = typed_informativeness_reward(messages=sample_messages)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bchen/home/reward-kit/reward_kit/typed_interface.py", line 86, in wrapper
    result = func(typed_messages, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bchen/home/reward-kit/examples/typed_reward_function_example.py", line 46, in typed_informativeness_reward
    Message(role=msg["role"], content=msg["content"])
  File "/home/bchen/miniconda3/lib/python3.12/typing.py", line 1140, in __call__
    result = self.__origin__(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/bchen/miniconda3/lib/python3.12/typing.py", line 480, in __call__
    raise TypeError(f"Cannot instantiate {self!r}")
TypeError: Cannot instantiate typing.Union
```

and in general for reward kit, I want to make sure the input is typed for the reward function, can you help me with that across my codebase? This will give the reward author more confidence that their code is correct.

In our typed_evaluator_example.py, things are correct

```

@reward_function
def evaluate(messages: List[Message], **kwargs) -> EvaluateResult:
```