# Goal
- clean up the code
- makes the type more correct

and make sure we use local venv to do this in case you need to install any dependencies


## Clean up the code
right now to author an evaluator, we need to make two rows of import

```
from reward_kit.models import Message, EvaluateResult, MetricResult
from reward_kit.typed_interface import reward_function
```

which is really ugly, please help me refactor the code so everything is just one row
o
## Make types more correct

Right now for messages we have

```
# Pydantic models for typed interface
class Message(BaseModel):
    """A message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
```

But I think we just accept the OpenAI message type, can you help me import from OpenAI instead as well as the OpenAI types?

Checking the OpenAI code, I think the base type is here

```
class CompletionCreateParamsBase(TypedDict, total=False):
    messages: Required[Iterable[ChatCompletionMessageParam]]
    """A list of messages comprising the conversation so far.

    Depending on the [model](https://platform.openai.com/docs/models) you use,
    different message types (modalities) are supported, like
    [text](https://platform.openai.com/docs/guides/text-generation),
    [images](https://platform.openai.com/docs/guides/vision), and
    [audio](https://platform.openai.com/docs/guides/audio).
    """

    model: Required[Union[str, ChatModel]]
    """Model ID used to generate the response, like `gpt-4o` or `o3`.

    OpenAI offers a wide range of models with different capabilities, performance
    characteristics, and price points. Refer to the
    [model guide](https://platform.openai.com/docs/models) to browse and compare
    available models.
    """

    audio: Optional[ChatCompletionAudioParam]
    """Parameters for audio output.

    Required when audio output is requested with `modalities: ["audio"]`.
    [Learn more](https://platform.openai.com/docs/guides/audio).
    """
```

under src/openai/types/chat/completion_create_params.py, but that is source code, we need to find the right type to import from OpenAI library

Also add unittest to make sure we are properly covered for all these different types