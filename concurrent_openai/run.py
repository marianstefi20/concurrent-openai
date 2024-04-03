from openai.types.chat import ChatCompletionMessageParam

from .openai_concurrent_manager import OpenAIConcurrentManager
from .openai_wrapper import OpenAIWrapper


async def process_completion_requests(
    prompts: list[list[ChatCompletionMessageParam]],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 100,
    max_concurrent_requests: int = 100,
    token_safety_margin: int = 10,
) -> list[dict | None]:
    openai_wrapper = OpenAIWrapper(
        model=model, temperature=temperature, max_tokens=max_tokens
    )
    completion_concurrency_manager = OpenAIConcurrentManager(
        openai_wrapper=openai_wrapper,
        semaphore_value=max_concurrent_requests,
        token_safety_margin=token_safety_margin,
    )
    return await completion_concurrency_manager.process_completion_requests(prompts)
