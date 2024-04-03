import logging

import backoff
import openai
import structlog
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .settings import settings

LOGGER = structlog.get_logger(__name__)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


class OpenAIWrapper:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        timeout: float = 180,
        temperature: float = 0.1,
        max_tokens: int = 150,
    ):
        self.model = model
        self.timeout = timeout
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError),
        max_time=20,
        logger=LOGGER,
    )
    async def get_completion(
        self, messages: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        response = await self.client.chat.completions.create(
            timeout=self.timeout,
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response

    @property
    def input_token_cost(self) -> float:
        return settings.OPENAI_MODEL_DETAILS[self.model]["input_token_cost"]

    @property
    def output_token_cost(self) -> float:
        return settings.OPENAI_MODEL_DETAILS[self.model]["output_token_cost"]
