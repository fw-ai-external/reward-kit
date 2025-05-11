# API Design for Composable Reward Functions

## Problem Statement

The current `@reward_function` decorator (defined in `reward_kit/typed_interface.py`) is designed to provide a consistent dictionary-based input/output interface for reward functions. While this is useful for standardization and serialization, it makes it inconvenient to compose reward functions.

The decorator takes a function that internally uses Pydantic models (`List[Message]` for input, `EvaluateResult` for output) and wraps it. The wrapper accepts `List[Dict]` as input and, crucially, converts the `EvaluateResult` object from the internal function back into a plain `Dict` using `model_dump()` before returning.

This final conversion to a dictionary means that rich type information is lost. If one decorated function's output is intended to be the input for another function that expects an `EvaluateResult` object (or another specific object type), this direct composition is not possible as a dictionary is received instead.

## Goal

Redesign the reward function API to allow for easier composition of reward functions, enabling them to pass richer data types (like `EvaluateResult` objects or other custom objects) between each other, while still providing an option for a dictionary-based interface for serialization or compatibility with external systems.

## Proposed Options

Here are several options for redesigning the API:

### Option 1: Dual-Mode Decorator

The `@reward_function` decorator could be modified to include a parameter that controls the return type.

```python
# In reward_kit/typed_interface.py
def reward_function(return_object: bool = False): # New parameter
    def decorator(func: EvaluateFunction) -> Union[DictEvaluateFunction, Callable[..., EvaluateResult]]:
        @wraps(func)
        def wrapper(
            messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
        ) -> Union[Dict[str, Any], EvaluateResult]:
            # ... (input coercion logic remains the same) ...
            typed_messages = _coerce_input_messages(messages)

            result_obj = func(typed_messages, **kwargs) # This should be an EvaluateResult or compatible

            # Coerce to EvaluateResult if it's a dict
            if not isinstance(result_obj, EvaluateResult):
                result_obj = _res_adapter.validate_python(result_obj)

            if return_object:
                return result_obj # Return the EvaluateResult object itself
            else:
                return result_obj.model_dump() # Current behavior: return a dict
        
        if return_object:
            return cast(Callable[..., EvaluateResult], wrapper)
        else:
            return cast(DictEvaluateFunction, wrapper)
    return decorator

# Usage
@reward_function(return_object=True)
def my_composable_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)

@reward_function() # Defaults to return_object=False
def my_dict_output_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)
```

**Pros:**
*   Single decorator, potentially less confusing than multiple decorators.
*   Flexible for the user.

**Cons:**
*   The decorator's return type becomes more complex (`Union`).
*   Slightly more verbose usage if `return_object=True` is common.
*   The decorator itself becomes a higher-order function (it returns the actual decorator).

### Option 2: Separate Decorators

Introduce a new decorator specifically for composable functions that return objects, keeping the existing one for dictionary outputs.

```python
# In reward_kit/typed_interface.py

# Existing @reward_function (returns dict)
# ... remains as is ...

# New decorator
def typed_reward_function(func: EvaluateFunction) -> Callable[..., EvaluateResult]:
    """
    Wraps an evaluate-style function, performing input coercion but returning 
    the direct EvaluateResult object (or compatible type from func).
    """
    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> EvaluateResult:
        typed_messages = _coerce_input_messages(messages) # Helper for input coercion
        
        result_obj = func(typed_messages, **kwargs)

        # Ensure it's an EvaluateResult or can be coerced
        if not isinstance(result_obj, EvaluateResult):
            # This step ensures the function still adheres to returning something
            # that can be understood as an EvaluateResult, even if it's an object.
            # If func is guaranteed to return EvaluateResult, this might be simplified.
            result_obj = _res_adapter.validate_python(result_obj) 
        
        return result_obj
    return cast(Callable[..., EvaluateResult], wrapper)

# Usage
@typed_reward_function
def my_composable_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)

@reward_function 
def my_dict_output_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)
```

**Pros:**
*   Clear separation of concerns. The decorator name indicates its behavior.
*   Simpler implementation for each decorator.
*   Type signature of each decorator is simpler.

**Cons:**
*   Users need to know about and choose between two decorators.

### Option 3: Accessing the Original/Typed Function via an Attribute

The existing `@reward_function` decorator could attach the underlying typed function (or a version that returns objects) as an attribute to the wrapped dictionary-returning function.

```python
# In reward_kit/typed_interface.py
def reward_function(func: EvaluateFunction) -> DictEvaluateFunction:
    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> Dict[str, Any]:
        # ... (current logic for input coercion and calling func) ...
        typed_messages = _coerce_input_messages(messages)
        result_obj = func(typed_messages, **kwargs)
        if not isinstance(result_obj, EvaluateResult):
            result_obj = _res_adapter.validate_python(result_obj)
        return result_obj.model_dump()

    # Create the object-returning version
    @wraps(func) # Preserve metadata of original func
    def object_returning_version(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> EvaluateResult:
        typed_messages = _coerce_input_messages(messages)
        result_obj = func(typed_messages, **kwargs)
        if not isinstance(result_obj, EvaluateResult):
            result_obj = _res_adapter.validate_python(result_obj)
        return result_obj
    
    wrapper.as_object = cast(Callable[..., EvaluateResult], object_returning_version)
    # or wrapper.raw_function = func (if func already handles coercion or is strictly typed)
    
    return cast(DictEvaluateFunction, wrapper)

# Usage
@reward_function
def my_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)

dict_result = my_reward_func(some_messages)
object_result = my_reward_func.as_object(some_messages) 
```

**Pros:**
*   Maintains backward compatibility for the default behavior.
*   Composition is possible by accessing the `.as_object` (or similar) attribute.

**Cons:**
*   Accessing an attribute for a different behavior might feel less clean or discoverable.
*   The type hinting for the decorated function and its attribute can be tricky.

### Option 4: Decorator Returns a Wrapper Object

The decorator could return an object that provides different methods for different return types.

```python
# In reward_kit/typed_interface.py
class RewardFunctionWrapper:
    def __init__(self, func: EvaluateFunction):
        self._typed_func = func # The original function expecting typed inputs
        # Potentially pre-validate or inspect func signature here

    def _call_internal(self, messages_input: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any) -> EvaluateResult:
        typed_messages = _coerce_input_messages(messages_input)
        result_obj = self._typed_func(typed_messages, **kwargs)
        if not isinstance(result_obj, EvaluateResult):
            result_obj = _res_adapter.validate_python(result_obj)
        return result_obj

    def __call__(self, messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any) -> Dict[str, Any]:
        """Default call returns a dictionary."""
        result_obj = self._call_internal(messages, **kwargs)
        return result_obj.model_dump()

    def call_object(self, messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any) -> EvaluateResult:
        """Call and return the EvaluateResult object."""
        return self._call_internal(messages, **kwargs)

def reward_function(func: EvaluateFunction) -> RewardFunctionWrapper:
    return RewardFunctionWrapper(func)

# Usage
@reward_function
def my_reward_func(messages: List[Message]) -> EvaluateResult:
    # ...
    return EvaluateResult(...)

dict_result = my_reward_func(some_messages) # Calls __call__
object_result = my_reward_func.call_object(some_messages)
```

**Pros:**
*   Clean and explicit API via methods on the wrapper object.
*   Good discoverability of different calling conventions.
*   The wrapper object can potentially hold more state or offer more utilities in the future.

**Cons:**
*   The decorated function is no longer a simple callable in the traditional sense for its default behavior (it's an object that *is* callable). This might affect introspection or how it's treated by some frameworks, though `__call__` makes it behave like a function.

### Option 5: Conditional Dumping Based on Function's Return Annotation (More Implicit)

Modify the existing decorator to inspect the wrapped function's return type annotation. If it's `EvaluateResult`, it could avoid dumping to dict. This is more implicit.

```python
# In reward_kit/typed_interface.py
import inspect

def reward_function(func: EvaluateFunction) -> Union[DictEvaluateFunction, Callable[..., EvaluateResult]]:
    @wraps(func)
    def wrapper(
        messages: Union[List[Dict[str, Any]], List[Message]], **kwargs: Any
    ) -> Union[Dict[str, Any], EvaluateResult]:
        typed_messages = _coerce_input_messages(messages)
        result_obj = func(typed_messages, **kwargs)

        # Ensure result_obj is an EvaluateResult instance
        if not isinstance(result_obj, EvaluateResult):
            result_obj = _res_adapter.validate_python(result_obj)

        # Check original function's return annotation
        sig = inspect.signature(func)
        if sig.return_annotation == EvaluateResult:
            return result_obj # Return as object
        else:
            return result_obj.model_dump() # Default to dict

    # This casting becomes tricky due to conditional return type
    # Needs careful thought on how to represent this to type checkers
    # For simplicity, one might assume it usually returns DictEvaluateFunction
    # and users wanting objects would type assert or ignore type errors.
    return cast(DictEvaluateFunction, wrapper) 
```

**Pros:**
*   Potentially "just works" for functions annotated to return `EvaluateResult`.

**Cons:**
*   Behavior is implicit and depends on type annotations, which might not always be present or accurate.
*   Harder to type correctly and for users to understand the exact return type without inspecting the wrapped function's annotations.
*   Could lead to unexpected behavior if annotations are changed.

## Recommendation

**Option 2 (Separate Decorators)** or **Option 4 (Decorator Returns a Wrapper Object)** seem like the cleanest and most explicit solutions.

*   **Option 2 (`typed_reward_function`)** is simple and makes the intent very clear through the decorator name. It's a common pattern to have different decorators for different behaviors.
*   **Option 4 (`RewardFunctionWrapper` object)** offers a very explicit API (`.call_object()`) and is perhaps the most "object-oriented". It also provides a natural place to add more functionality to the wrapped reward function in the future.

**Option 1 (Dual-Mode Decorator)** is also viable but makes the decorator itself a bit more complex (being a function that returns a decorator).

A common helper function `_coerce_input_messages` should be used by any chosen solution to handle the conversion from `List[Dict]` to `List[Message]` to avoid code duplication. Similarly, a `_ensure_evaluate_result` helper could be used.

Let's assume `_coerce_input_messages` would look something like this (extracted from the current `typed_interface.py`):
```python
def _coerce_input_messages(messages_input: Union[List[Dict[str, Any]], List[Message]]) -> List[Message]:
    try:
        typed_messages = []
        for msg in messages_input:
            if isinstance(msg, Message):
                typed_messages.append(msg)
            else:
                # Simplified for brevity, full validation from typed_interface.py
                typed_messages.append(Message(**msg)) 
        return typed_messages
    except Exception as err:
        raise ValueError(f"Input messages failed validation:\n{err}") from None
```

Further discussion is needed to decide on the best path forward. The primary trade-off is between explicitness, simplicity, and the number of new symbols introduced.
