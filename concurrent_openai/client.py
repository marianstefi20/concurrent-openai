import asyncio
import os
from typing import Any, Optional

import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .models import ConcurrentCompletionResponse
from .rate_limiter import RateLimiter
from .utils import count_total_tokens

LOGGER = structlog.get_logger(__name__)

load_dotenv()


class ConcurrentOpenAI:
    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent_requests: int = 100,
        token_safety_margin: int = 100,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        input_token_cost: Optional[float] = None,
        output_token_cost: Optional[float] = None,
        **client_options: Any,
    ):
        """
        Initialize a concurrent OpenAI client.

        Args:
            api_key: OpenAI API key
            max_concurrent_requests: Maximum number of concurrent requests
            token_safety_margin: Safety margin for token estimation
            requests_per_minute: Maximum requests per minute (optional)
            tokens_per_minute: Maximum tokens per minute (optional)
            input_token_cost: Cost per input token (optional)
            output_token_cost: Cost per output token (optional)
            **client_options: Additional options passed to AsyncOpenAI client
        """
        # set api key from env in case it's not provided
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = AsyncOpenAI(api_key=api_key, **client_options)
        self.token_safety_margin = token_safety_margin
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost

        self.request_limiter = (
            RateLimiter(
                capacity=requests_per_minute,
                fill_rate=requests_per_minute / 60,
                minimum_spacing=1 / (requests_per_minute / 60),
            )
            if requests_per_minute
            else None
        )

        self.token_limiter = (
            RateLimiter(
                capacity=tokens_per_minute,
                fill_rate=tokens_per_minute / 60,
                minimum_spacing=1 / (tokens_per_minute / 60),
            )
            if tokens_per_minute
            else None
        )

    async def create(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "gpt-3.5-turbo",
        **kwargs: Any,
    ) -> ConcurrentCompletionResponse:
        """
        Create a completion with rate limiting and concurrency control.
        Accepts all OpenAI chat completion parameters.
        """
        async with self.semaphore:
            # Calculate token estimation
            estimated_total_tokens = (
                count_total_tokens(messages, tools, model) + self.token_safety_margin
            )

            # Apply rate limiting if enabled
            if self.request_limiter:
                await self.request_limiter.acquire(1)

            if self.token_limiter:
                await self.token_limiter.acquire(estimated_total_tokens)

            try:
                response = await self.client.chat.completions.create(
                    messages=messages, model=model, **kwargs  # type: ignore
                )

                if response.usage is None:
                    LOGGER.error("Missing usage information in response", response=response)
                    return ConcurrentCompletionResponse(
                        openai_response=response,
                        estimated_total_tokens=estimated_total_tokens,
                        error="Missing usage information in response",
                    )

                input_cost = output_cost = total_cost = 0.0

                # Calculate costs if token costs are provided
                if self.input_token_cost and self.output_token_cost:
                    input_cost = response.usage.prompt_tokens * self.input_token_cost
                    output_cost = response.usage.completion_tokens * self.output_token_cost

                return ConcurrentCompletionResponse(
                    openai_response=response,
                    estimated_total_tokens=estimated_total_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                )

            except Exception as e:
                LOGGER.error(
                    "Error processing completion request",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return ConcurrentCompletionResponse(
                    estimated_total_tokens=estimated_total_tokens, error=str(e)
                )

    async def create_many(
        self, messages_list: list[list[dict[str, Any]]], **kwargs: Any
    ) -> list[ConcurrentCompletionResponse]:
        """Create multiple completions concurrently."""
        return await asyncio.gather(
            *(self.create(messages=messages, **kwargs) for messages in messages_list)
        )
