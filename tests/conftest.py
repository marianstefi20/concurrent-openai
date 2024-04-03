import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


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
