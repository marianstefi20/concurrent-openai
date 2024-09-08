import base64
import math
import struct
from typing import Any

import structlog
import tiktoken

LOGGER = structlog.get_logger(__name__)

MODEL_PREFIX_TO_TOKENS = {
    "gpt-3.5": {"tokens_per_message": 3, "tokens_per_name": 1},
    "gpt-4": {"tokens_per_message": 3, "tokens_per_name": 1},
    "gpt-4o": {"tokens_per_message": 3, "tokens_per_name": 1},
}


def num_tokens_from_messages(messages: list[dict], model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.
    Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    to also work with GPT-4 vision.
    """
    encoding = get_encoding(model)
    model_tokens = get_model_tokens(model)
    tokens_per_message = model_tokens["tokens_per_message"]
    tokens_per_name = model_tokens["tokens_per_name"]

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += _count_tokens_for_message_part(key, value, encoding)
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_model_tokens(model: str) -> dict[str, int]:
    for prefix, tokens in MODEL_PREFIX_TO_TOKENS.items():
        if model.startswith(prefix):
            return tokens
    raise NotImplementedError(
        f"num_tokens_from_messages() is not implemented for model {model}."
    )


def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        LOGGER.warning("Model not found. Using cl100k_base encoding.", model=model)
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens_for_message_part(
    key: str, value: Any, encoding: tiktoken.Encoding
) -> int:
    if isinstance(value, str):
        return len(encoding.encode(value))
    elif isinstance(value, list):
        return sum(_count_tokens_for_list_item(item, encoding) for item in value)
    else:
        LOGGER.error(f"Could not encode unsupported message key type: {type(key)}")
        return 0


def _count_tokens_for_list_item(
    item: dict[str, Any], encoding: tiktoken.Encoding
) -> int:
    num_tokens = len(encoding.encode(item["type"]))
    if item["type"] == "text":
        num_tokens += len(encoding.encode(item["text"]))
    elif item["type"] == "image_url":
        width, height = get_png_dimensions(item["image_url"]["url"])
        num_tokens += num_tokens_for_image(width, height)
    else:
        LOGGER.error(f"Could not encode unsupported message value type: {type(item)}")
    return num_tokens


def get_png_dimensions(base64_str: str) -> tuple[int, int]:
    png_prefix = "data:image/png;base64,"
    if not base64_str.startswith(png_prefix):
        raise ValueError("Base64 string is not a PNG image.")
    base64_str = base64_str.replace(png_prefix, "")
    decoded_bytes = base64.b64decode(base64_str[: 33 * 4 // 3], validate=True)
    width, height = struct.unpack(">II", decoded_bytes[16:24])
    return width, height


def num_tokens_for_image(width: int, height: int, low_resolution: bool = False) -> int:
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
