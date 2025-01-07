from dataclasses import dataclass

from openai.types.chat import ChatCompletion


@dataclass
class ConcurrentCompletionResponse:
    """
    Wrapper around OpenAI's response with concurrent-specific information.
    """

    openai_response: ChatCompletion | None = None

    # Library-specific metrics
    estimated_total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    # Error handling
    error: str | None = None

    @property
    def content(self) -> str | None:
        """Convenience accessor for the response content."""
        if self.openai_response and self.openai_response.choices:
            return self.openai_response.choices[0].message.content
        return None

    @property
    def is_success(self) -> bool:
        """Convenience accessor for request success."""
        return self.error is None

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class ModelTokenSettings:
    # Message-related settings
    tokens_per_message: int
    tokens_per_name: int

    # Function-related settings
    tokens_per_function: int
    tokens_per_property: int
    tokens_per_property_key: int
    tokens_per_enum_start: int
    tokens_per_enum_item: int
    tokens_per_function_end: int
