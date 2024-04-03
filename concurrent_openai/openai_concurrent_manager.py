import asyncio

import openai
import structlog
from openai.types.chat import ChatCompletionMessageParam

from .openai_wrapper import OpenAIWrapper
from .rate_limiter import RateLimiter
from .settings import settings
from .utils import num_tokens_from_messages

LOGGER = structlog.get_logger(__name__)


class OpenAIConcurrentManager:
    def __init__(
        self,
        openai_wrapper: OpenAIWrapper,
        rate_limiter: RateLimiter | None = None,
        semaphore_value: int = 100,
        token_safety_margin: int = settings.TOKEN_SAFETY_MARGIN,
    ):
        if rate_limiter is None:
            rate_limiter = RateLimiter(openai_wrapper.model)

        self.openai_wrapper = openai_wrapper
        self.rate_limiter = rate_limiter
        self.token_safety_margin = token_safety_margin
        self.semaphore = asyncio.Semaphore(semaphore_value)

    async def process_completion_request(
        self, messages: list[ChatCompletionMessageParam]
    ) -> None | dict:
        async with self.semaphore:
            token_estimation = (
                num_tokens_from_messages(
                    [dict(m) for m in messages], self.openai_wrapper.model
                )
                + self.token_safety_margin
            )
            await self.rate_limiter.acquire(token_estimation)
            try:
                response = await self.openai_wrapper.get_completion(messages)
            except openai.OpenAIError as exc:
                LOGGER.error(
                    "HTTP error while processing completion request",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                return

            if response.usage is None:
                LOGGER.error("Missing usage information in response", response=response)
                return

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            consumed_tokens = response.usage.total_tokens
            total_cost = (
                prompt_tokens * self.openai_wrapper.input_token_cost
                + completion_tokens * self.openai_wrapper.output_token_cost
            )
            await self.rate_limiter.release(consumed_tokens)
            return {
                "response": response,
                "estimated_prompt_tokens": token_estimation,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_cost": total_cost,
            }

    async def process_completion_requests(
        self, requests: list[list[ChatCompletionMessageParam]]
    ) -> list[dict | None]:
        return await asyncio.gather(
            *[self.process_completion_request(request) for request in requests]
        )
