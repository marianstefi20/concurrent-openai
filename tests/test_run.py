from unittest.mock import AsyncMock, patch

import pytest

from concurrent_openai import process_completion_requests


@pytest.mark.asyncio
async def test_process_completion_requests_cleanup(mocked_openai_wrapper_response):
    prompts = [[{"role": "user", "content": "mock message"}]]
    model = "test-gpt-4"

    with (
        patch("concurrent_openai.run.OpenAIWrapper") as MockOpenAIWrapper,
        patch(
            "concurrent_openai.openai_concurrent_manager.RateLimiter"
        ) as MockRateLimiter,
    ):
        mock_openai_wrapper_instance = MockOpenAIWrapper.return_value
        mock_openai_wrapper_instance.model = model
        mock_openai_wrapper_instance.get_completion = AsyncMock(
            return_value=mocked_openai_wrapper_response
        )

        mock_rate_limiter_instance = MockRateLimiter.return_value
        mock_rate_limiter_instance.cleanup = AsyncMock()
        mock_rate_limiter_instance.acquire = AsyncMock()
        mock_rate_limiter_instance.release = AsyncMock()

        await process_completion_requests(
            prompts=prompts,  # type: ignore
            model=model,
            temperature=0.1,
            max_tokens=100,
            max_concurrent_requests=100,
            token_safety_margin=10,
        )

        mock_rate_limiter_instance.cleanup.assert_awaited_once()
