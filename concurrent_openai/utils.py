import base64
import math
import struct

import structlog
import tiktoken

LOGGER = structlog.get_logger(__name__)


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


def get_png_dimensions(base64_str: str) -> tuple[int, int]:
    png_prefix = "data:image/png;base64,"
    if not base64_str.startswith(png_prefix):
        raise ValueError("Base64 string is not a PNG image.")
    base64_str = base64_str.replace(png_prefix, "")
    decoded_bytes = base64.b64decode(base64_str[: 33 * 4 // 3], validate=True)
    width, height = struct.unpack(">II", decoded_bytes[16:24])
    return width, height


def num_tokens_from_messages(messages: list[dict], model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.
    Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    to also work with GPT-4 vision.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        LOGGER.warning("Model not found. Using cl100k_base encoding.", model=model)
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        LOGGER.info(
            "gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        LOGGER.info(
            "gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                for item in value:
                    num_tokens += len(encoding.encode(item["type"]))
                    if item["type"] == "text":
                        num_tokens += len(encoding.encode(item["text"]))
                    elif item["type"] == "image_url":
                        width, height = get_png_dimensions(item["image_url"]["url"])
                        num_tokens += num_tokens_for_image(width, height)
                    else:
                        raise LOGGER.error(
                            f"Could not encode unsupported message value type: {type(value)}"
                        )
            else:
                raise LOGGER.error(
                    f"Could not encode unsupported message key type: {type(key)}"
                )

            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
