import base64
import math
import struct
from typing import Any

import structlog
import tiktoken

from concurrent_openai.models import ModelTokenSettings

LOGGER = structlog.get_logger(__name__)


MODEL_SETTINGS: dict[str, ModelTokenSettings] = {
    "gpt-3.5": ModelTokenSettings(
        tokens_per_message=3,
        tokens_per_name=1,
        tokens_per_function=10,
        tokens_per_property=3,
        tokens_per_property_key=3,
        tokens_per_enum_start=-3,
        tokens_per_enum_item=3,
        tokens_per_function_end=12,
    ),
    "gpt-4": ModelTokenSettings(
        tokens_per_message=3,
        tokens_per_name=1,
        tokens_per_function=10,
        tokens_per_property=3,
        tokens_per_property_key=3,
        tokens_per_enum_start=-3,
        tokens_per_enum_item=3,
        tokens_per_function_end=12,
    ),
    "gpt-4o": ModelTokenSettings(
        tokens_per_message=3,
        tokens_per_name=1,
        tokens_per_function=7,
        tokens_per_property=3,
        tokens_per_property_key=3,
        tokens_per_enum_start=-3,
        tokens_per_enum_item=3,
        tokens_per_function_end=12,
    ),
}


def count_total_tokens(messages: list[dict], tools: list[dict] | None, model: str) -> int:
    return count_message_tokens(messages, model) + count_function_tokens(tools, model)


def count_message_tokens(messages: list[dict], model: str = "gpt-3.5-turbo") -> int:
    """
    Return the number of tokens used by a list of messages.
    Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    Args:
        messages: List of message dictionaries with role and content
        model: The model to count tokens for

    Returns:
        int: Number of tokens in the messages
    """
    encoding = get_encoding(model)
    settings = get_model_settings(model)

    num_tokens = 0
    for message in messages:
        num_tokens += settings.tokens_per_message
        for key, value in message.items():
            num_tokens += _count_tokens_for_message_part(key, value, encoding)
            if key == "name":
                num_tokens += settings.tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_function_tokens(tools: list[dict] | None, model: str) -> int:
    """
    Return the number of tokens used by a list of functions.
    Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    Args:
        tools: List of function dictionaries with name and description
        model: The model to count tokens for

    Returns:
        int: Number of tokens in the messages
    """

    if not tools:
        return 0

    settings = get_model_settings(model)
    encoding = get_encoding(model)

    func_token_count = 0
    if len(tools) > 0:
        for f in tools:
            func_token_count += (
                settings.tokens_per_function
            )  # Add tokens for start of each function
            function = f["function"]
            f_name = function["name"]
            f_desc = function["description"]
            if f_desc.endswith("."):
                f_desc = f_desc[:-1]
            line = f_name + ":" + f_desc
            func_token_count += len(
                encoding.encode(line)
            )  # Add tokens for set name and description
            if len(function["parameters"]["properties"]) > 0:
                func_token_count += (
                    settings.tokens_per_property
                )  # Add tokens for start of each property
                for key in list(function["parameters"]["properties"].keys()):
                    func_token_count += (
                        settings.tokens_per_property_key
                    )  # Add tokens for each set property
                    p_name = key
                    p_type = function["parameters"]["properties"][key]["type"]
                    p_desc = function["parameters"]["properties"][key]["description"]
                    if "enum" in function["parameters"]["properties"][key].keys():
                        func_token_count += (
                            settings.tokens_per_enum_start
                        )  # Add tokens if property has enum list
                        for item in function["parameters"]["properties"][key]["enum"]:
                            func_token_count += settings.tokens_per_enum_item
                            func_token_count += len(encoding.encode(item))
                    if p_desc.endswith("."):
                        p_desc = p_desc[:-1]
                    line = f"{p_name}:{p_type}:{p_desc}"
                    func_token_count += len(encoding.encode(line))
        func_token_count += settings.tokens_per_function_end

    return func_token_count


def get_model_settings(model: str) -> ModelTokenSettings:
    """Get the token settings for a given model."""
    for prefix, settings in MODEL_SETTINGS.items():
        if model.startswith(prefix):
            return settings

    # Fallback to gpt-4o settings
    LOGGER.warning("Model not found. Using gpt-4o settings.", model=model)
    return MODEL_SETTINGS["gpt-4o"]


def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        LOGGER.warning("Model not found. Using o200k_base encoding.", model=model)
        return tiktoken.get_encoding("o200k_base")


def get_png_dimensions(base64_str: str) -> tuple[int, int]:
    """Extract width and height from a base64-encoded PNG image.

    Args:
        base64_str: Base64-encoded PNG image string with 'data:image/png;base64,' prefix

    Returns:
        tuple[int, int]: Width and height of the image. Returns (0, 0) if invalid format.
    """
    png_prefix = "data:image/png;base64,"
    if not base64_str.startswith(png_prefix):
        LOGGER.warning(
            "Invalid PNG image format",
            expected_prefix=png_prefix,
            received_prefix=base64_str[: min(len(png_prefix), len(base64_str))],
        )
        return 0, 0

    base64_str = base64_str.replace(png_prefix, "")
    try:
        decoded_bytes = base64.b64decode(base64_str[: 33 * 4 // 3], validate=True)
        width, height = struct.unpack(">II", decoded_bytes[16:24])
        return width, height
    except (base64.binascii.Error, struct.error) as e:  # type: ignore
        LOGGER.warning("Failed to decode PNG dimensions", error=str(e))
        return 0, 0


def _count_tokens_for_message_part(key: str, value: Any, encoding: tiktoken.Encoding) -> int:
    if isinstance(value, str):
        return len(encoding.encode(value))
    elif isinstance(value, list):
        return sum(_count_tokens_for_list_item(item, encoding) for item in value)
    else:
        LOGGER.error(f"Could not encode unsupported message key type: {type(key)}")
        return 0


def _count_tokens_for_list_item(item: dict[str, Any], encoding: tiktoken.Encoding) -> int:
    num_tokens = len(encoding.encode(item["type"]))
    if item["type"] == "text":
        num_tokens += len(encoding.encode(item["text"]))
    elif item["type"] == "image_url":
        width, height = get_png_dimensions(item["image_url"]["url"])
        num_tokens += _count_image_tokens(width, height)
    else:
        LOGGER.error(f"Could not encode unsupported message value type: {type(item)}")
    return num_tokens


def _count_image_tokens(width: int, height: int, low_resolution: bool = False) -> int:
    """
    Calculate the number of tokens for an image.
    Based on the https://platform.openai.com/docs/guides/vision#calculating-costs

    Args:
        width: The width of the image
        height: The height of the image
        low_resolution: Whether the image is low resolution

    Returns:
        int: The number of tokens for the image
    """
    if width <= 0 or height <= 0:
        LOGGER.warning("Invalid image dimensions", width=width, height=height)
        return 0

    BASE_TOKENS = 85
    TILE_TOKENS = 170
    TILE_LENGTH = 512

    MAX_LENGTH = 2048
    MEDIUM_LENGTH = 768

    if low_resolution:
        return BASE_TOKENS

    if max(width, height) > MAX_LENGTH:
        ratio = MAX_LENGTH / max(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    if min(width, height) > MEDIUM_LENGTH:
        ratio = MEDIUM_LENGTH / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)

    num_tiles = math.ceil(width / TILE_LENGTH) * math.ceil(height / TILE_LENGTH)
    return BASE_TOKENS + num_tiles * TILE_TOKENS
