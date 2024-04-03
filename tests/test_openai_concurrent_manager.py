import asyncio
from unittest import mock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from concurrent_openai.openai_concurrent_manager import OpenAIConcurrentManager
from concurrent_openai.openai_wrapper import OpenAIWrapper
from concurrent_openai.rate_limiter import RateLimiter


@pytest.fixture
def mocked_message():
    return {
        "role": "user",
        "content": "mock message",
    }


@pytest.fixture
def mocked_openai_wrapper_response():
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
@mock.patch(
    "concurrent_openai.openai_concurrent_manager.settings.TOKEN_SAFETY_MARGIN", 10
)
@mock.patch(
    "concurrent_openai.openai_concurrent_manager.settings.OPENAI_MODEL_DETAILS",
    {
        "gpt-4-0613": {
            "rpm": 600,
            "tkm": 6000,
            "input_token_cost": 1,
            "output_token_cost": 3,
        }
    },
)
@pytest.mark.parametrize("semaphore_value", [1, 2, 3])
async def test_process_completion_request(
    mocked_openai_wrapper_response, semaphore_value
):
    completion_delay = 0.01
    nr_of_requests = 6

    async def mock_get_completion(_, **kwargs):
        await asyncio.sleep(completion_delay)
        return mocked_openai_wrapper_response

    mocked_openai_wrapper = mock.AsyncMock(spec=OpenAIWrapper)
    mocked_openai_wrapper.model = "gpt-4-0613"
    mocked_openai_wrapper.get_completion = mock.AsyncMock(
        side_effect=mock_get_completion
    )

    rate_limiter = RateLimiter("gpt-4-0613")
    rate_limiter.refill_interval = 0.02

    manager = OpenAIConcurrentManager(
        openai_wrapper=mocked_openai_wrapper,
        rate_limiter=rate_limiter,
        semaphore_value=semaphore_value,
    )

    requests = [
        [{"role": "user", "content": "mock message"}] for _ in range(nr_of_requests)
    ]

    start_time = asyncio.get_event_loop().time()
    await manager.process_completion_requests(requests)  # type: ignore
    end_time = asyncio.get_event_loop().time()
    elapsed_time = end_time - start_time

    # The expected time is approximately the completion_delay times the number
    # of batches of requests divided by semaphore_value
    expected_batches = nr_of_requests // semaphore_value
    expected_time = completion_delay * expected_batches
    assert (
        elapsed_time >= expected_time
    ), f"Elapsed time is less than expected for semaphore value {semaphore_value}"

    await rate_limiter.cleanup()
