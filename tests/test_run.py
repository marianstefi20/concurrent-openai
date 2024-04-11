import os
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from concurrent_openai import process_completion_requests

load_dotenv()


@pytest.mark.asyncio
async def test_process_completion_requests_cleanup(mocked_chat_completion):
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
            return_value=mocked_chat_completion
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


@pytest.mark.skipif(
    not os.getenv("ENABLE_COSTLY_TESTS") == "1", reason="ENABLE_COSTLY_TESTS is not '1'"
)
@pytest.mark.asyncio
async def test_process_complection_requests_vision(
    base64_sunglasses_image, mocked_process_completion_requests_response
):
    prompts = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in the image? Be very succint (just a couple of words).",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_sunglasses_image,
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
    ]
    responses = await process_completion_requests(
        prompts=prompts,  # type: ignore
        model="gpt-4-1106-vision-preview",
        temperature=0.05,
        max_tokens=100,
        max_concurrent_requests=100,
        token_safety_margin=10,
    )
    completion = responses[0]
    if not completion:
        assert False, "No completion was returned."

    assert (
        completion["response"].choices[0].message.content
        == mocked_process_completion_requests_response["response"]
        .choices[0]
        .message.content
    )
    assert (
        completion["response"].usage
        == mocked_process_completion_requests_response["response"].usage
    )
    assert (
        completion["estimated_prompt_tokens"]
        == mocked_process_completion_requests_response["estimated_prompt_tokens"]
    )
    assert (
        completion["prompt_tokens"]
        == mocked_process_completion_requests_response["prompt_tokens"]
    )
    assert (
        completion["completion_tokens"]
        == mocked_process_completion_requests_response["completion_tokens"]
    )
    assert (
        pytest.approx(completion["total_cost"], 0.0001)
        == mocked_process_completion_requests_response["total_cost"]
    )
