from dataclasses import dataclass

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
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


@dataclass
class CompletionResponse:
    api_response: ChatCompletion | None = None
    answer: str | None = None
    estimated_prompt_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    conversation_id: str = ""
    error_message: str = ""
