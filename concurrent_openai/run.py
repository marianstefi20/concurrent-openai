from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.completion_create_params import ResponseFormat

from .openai_concurrent_manager import OpenAIConcurrentManager
from .openai_wrapper import OpenAIWrapper
from .types import CompletionRequest, CompletionResponse


async def process_completion_requests(
    prompts: list[CompletionRequest],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 100,
    max_concurrent_requests: int = 100,
    token_safety_margin: int = 10,
    response_format: ResponseFormat | NotGiven = NOT_GIVEN,
) -> list[CompletionResponse]:
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


async def get_vision_response(
    prompt: str, base64_images: list[str], **kwargs
) -> CompletionResponse:
    responses = await process_completion_requests(
        prompts=[
            CompletionRequest(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_image},
                                }
                                for base64_image in base64_images
                            ],
                        ],
                    }
                ]
            )
        ],
        **kwargs,
    )
    return responses[0]


async def get_responses(prompts: list[str], **kwargs) -> list[CompletionResponse]:
    return await process_completion_requests(
        prompts=[
            CompletionRequest(messages=[{"role": "user", "content": prompt}])
            for prompt in prompts
        ],
        **kwargs,
    )
