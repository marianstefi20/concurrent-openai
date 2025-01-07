import pytest

from concurrent_openai.utils import (
    _count_image_tokens,
    count_function_tokens,
    count_message_tokens,
    count_total_tokens,
    get_png_dimensions,
)


@pytest.mark.parametrize(
    "width, height, low_resolution, expected",
    [
        (100, 100, True, 85),
        (1000, 1000, True, 85),
        (128, 128, False, 255),
        (256, 128, False, 255),
        (512, 128, False, 255),
        (513, 128, False, 425),
        (513, 513, False, 765),
        (760, 760, False, 765),
        (768, 768, False, 765),
        (1024, 1024, False, 765),
        (1530, 768, False, 1105),
        (1900, 1000, False, 1105),
        (2049, 784, False, 1445),
        (2100, 784, False, 1445),
        (3100, 784, False, 1445),
        (5000, 784, False, 765),
        (1900, 5000, False, 1445),
    ],
)
def test_num_tokens_for_image(width, height, low_resolution, expected):
    assert _count_image_tokens(width, height, low_resolution) == expected


def test_get_png_dimensions(base64_sunglasses_image):
    width, height = get_png_dimensions(base64_sunglasses_image)
    assert width == height == 20


@pytest.mark.parametrize(
    "model, actual_prompt_tokens",
    [
        ("gpt-4", 151),
        ("gpt-4o", 152),
        ("gpt-3.5", 151),
        ("gpt-3.5-turbo", 151),
    ],
)
def test_count_message_tokens_1(conversation1, model, actual_prompt_tokens):
    assert count_message_tokens(conversation1, model=model) == actual_prompt_tokens


@pytest.mark.parametrize(
    "model, actual_prompt_tokens",
    [
        ("gpt-4", 179),
        ("gpt-4o", 179),
        ("gpt-3.5", 179),
        ("gpt-3.5-turbo", 179),
    ],
)
def test_count_message_tokens_2(conversation2, model, actual_prompt_tokens):
    assert count_message_tokens(conversation2, model=model) == actual_prompt_tokens


def test_count_total_tokens_without_functions(conversation1):
    # Test counting tokens without any functions
    total_tokens = count_total_tokens(conversation1, tools=None, model="gpt-4o")
    # Must match test_count_message_tokens_1
    assert total_tokens == 152


def test_count_total_tokens_with_functions(conversation1):
    # Test counting tokens with functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to return the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    total_tokens = count_total_tokens(conversation1, tools=tools, model="gpt-4o")
    message_tokens = count_message_tokens(conversation1, model="gpt-4o")
    function_tokens = count_function_tokens(tools, model="gpt-4o")
    assert total_tokens == message_tokens + function_tokens


@pytest.mark.parametrize(
    "tools,model,expected_tokens",
    [
        # Empty functions list
        ([], "gpt-4", 0),
        # Simple function without parameters
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "gpt-4",
            27,
        ),
        # Function with string parameter
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Echo a message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string", "description": "Message to echo"}
                            },
                        },
                    },
                }
            ],
            "gpt-4",
            39,
        ),
        # Function with enum
        (
            [
                {
                    "type": "function",
                    "function": {
                        "name": "set_color",
                        "description": "Set the color",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "color": {
                                    "type": "string",
                                    "description": "The color to set",
                                    "enum": ["red", "green", "blue"],
                                }
                            },
                        },
                    },
                }
            ],
            "gpt-4",
            48,
        ),
    ],
)
def test_count_function_tokens(tools, model, expected_tokens):
    assert count_function_tokens(tools, model) == expected_tokens
