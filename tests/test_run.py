import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from concurrent_openai import CompletionRequest, process_completion_requests

load_dotenv()


@pytest.mark.asyncio
async def test_process_completion_requests_cleanup(mocked_chat_completion):
    prompts = [
        CompletionRequest(messages=[{"role": "user", "content": "mock message"}])
    ]
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
        CompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in the image? Reply with `Emoji with sunglasses.` or `Emoji without sunglasses.`.",
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
        )
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


@pytest.mark.skipif(
    not os.getenv("ENABLE_COSTLY_TESTS") == "1", reason="ENABLE_COSTLY_TESTS is not '1'"
)
@pytest.mark.asyncio
async def test_process_completion_requests_vision_with_tools(base64_sunglasses_image):
    prompts = [
        CompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert extration tool. Please extract the objects in the image using the available tools",
                },
                {
                    "role": "user",
                    "content": "What is in the image found at https://foodstyles.com/test_image?",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "image_parser",
                        "description": "Parse an image and return the objects in it.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_url": {
                                    "type": "string",
                                    "description": "The URL of the image to be parsed.",
                                }
                            },
                            "required": ["image_url"],
                        },
                    },
                }
            ],
        )
    ]

    responses = await process_completion_requests(
        prompts=prompts,  # type: ignore
        model="gpt-4",
        temperature=0.05,
        max_tokens=100,
        max_concurrent_requests=100,
        token_safety_margin=10,
    )

    completion = responses[0]
    if not completion:
        assert False, "No completion was returned."

    response_message = completion["response"].choices[0].message
    tool_calls = response_message.tool_calls
    if not tool_calls:
        assert False, "No tool calls were returned."

    tool_call = tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    assert function_name == "image_parser"
    assert function_args["image_url"] == "https://foodstyles.com/test_image"
