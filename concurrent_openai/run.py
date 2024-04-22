from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.completion_create_params import ResponseFormat

from .openai_concurrent_manager import OpenAIConcurrentManager
from .openai_wrapper import OpenAIWrapper
from .types import CompletionRequest


async def process_completion_requests(
    prompts: list[CompletionRequest],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 100,
    max_concurrent_requests: int = 100,
    token_safety_margin: int = 10,
    response_format: ResponseFormat | NotGiven = NOT_GIVEN,
) -> list[dict | None]:
    openai_wrapper = OpenAIWrapper(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    async with OpenAIConcurrentManager(
        openai_wrapper=openai_wrapper,
        semaphore_value=max_concurrent_requests,
        token_safety_margin=token_safety_margin,
    ) as completion_concurrency_manager:
        return await completion_concurrency_manager.process_completion_requests(prompts)
