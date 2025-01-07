import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from concurrent_openai.client import ConcurrentOpenAI

load_dotenv()


@pytest.fixture
def mocked_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-99XUGZR68HIAcvljfTyb5FYAxxtJH",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="Hello! How can I assist you today?",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1712060704,
        model="gpt-4-0613",
        object="chat.completion",
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=9, prompt_tokens=10, total_tokens=19),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("semaphore_value", [1, 2, 3])
async def test_concurrent_requests(semaphore_value, mocked_chat_completion):
    """Test that concurrent requests are properly handled with different semaphore values."""
    completion_delay = 0.01
    nr_of_requests = 6

    # Create a properly structured mock
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_chat = AsyncMock()
    mock_completions = AsyncMock()

    # Add delay to simulate API call
    async def delayed_response(*args, **kwargs):
        await asyncio.sleep(completion_delay)
        return mocked_chat_completion

    # Structure the mock to match AsyncOpenAI's structure
    mock_completions.create = AsyncMock(side_effect=delayed_response)
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    with patch("concurrent_openai.client.AsyncOpenAI", return_value=mock_client):
        client = ConcurrentOpenAI(
            api_key="test-key",
            max_concurrent_requests=semaphore_value,
            requests_per_minute=600,
            tokens_per_minute=6000,
        )

        messages_list = [
            [{"role": "user", "content": f"mock message {i}"}] for i in range(nr_of_requests)
        ]

        start_time = asyncio.get_event_loop().time()
        responses = await client.create_many(messages_list=messages_list)
        end_time = asyncio.get_event_loop().time()
        elapsed_time = end_time - start_time

        # Verify all requests were successful
        assert len(responses) == nr_of_requests
        assert all(response.is_success for response in responses)

        # The expected time is approximately the completion_delay times the number
        # of batches of requests divided by semaphore_value
        expected_batches = nr_of_requests / semaphore_value
        expected_time = completion_delay * expected_batches

        # Allow for some timing variance
        assert elapsed_time >= expected_time * 0.9, (
            f"Elapsed time ({elapsed_time:.2f}s) is less than expected ({expected_time:.2f}s) "
            f"for semaphore value {semaphore_value}"
        )

        await client.__aexit__(None, None, None)


@pytest.mark.skipif(
    not os.getenv("ENABLE_COSTLY_TESTS") == "1", reason="ENABLE_COSTLY_TESTS is not '1'"
)
@pytest.mark.asyncio
async def test_vision_request(base64_sunglasses_image):
    """Test vision model request with actual API call."""
    client = ConcurrentOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_concurrent_requests=1,
    )

    messages = [
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

    response = await client.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=100,
        temperature=0.05,
        tools=None,
    )

    assert response.is_success
    assert response.content == "Emoji with sunglasses."
    assert response.openai_response is not None
    assert response.openai_response.usage is not None
    assert response.openai_response.usage.prompt_tokens > 0
    assert response.openai_response.usage.completion_tokens > 0

    await client.__aexit__(None, None, None)
