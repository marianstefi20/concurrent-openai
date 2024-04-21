from dataclasses import dataclass

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)


@dataclass
class CompletionRequest:
    messages: list[ChatCompletionMessageParam]
    seed: int | NotGiven | None = NOT_GIVEN
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
