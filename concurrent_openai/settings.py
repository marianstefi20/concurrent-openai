from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""

    # This is a margin added when the token estimation is calculated to avoid underestimating the number of tokens
    # It should always be larger than the maximum number of tokens that is generated in a completion
    TOKEN_SAFETY_MARGIN: int = 100

    # Rate limits for different models, change these values according to your OpenAI plan
    OPENAI_MODEL_DETAILS: dict[str, dict[str, float]] = {
        "gpt-4o": {
            "rpm": 10_000,
            "tkm": 30_000_000,
            "input_token_cost": 5 / 1_000_000,
            "output_token_cost": 15 / 1_000_000,
            "max_tokens": 4096,
        },
        "gpt-4o-mini": {
            "rpm": 30_000,
            "tkm": 150_000_000,
            "input_token_cost": 0.150 / 1_000_000,
            "output_token_cost": 0.6 / 1_000_000,
            "max_tokens": 16384,
        },
        "gpt-4o-2024-08-06": {
            "rpm": 10_000,
            "tkm": 30_000_000,
            "input_token_cost": 2.5 / 1_000_000,
            "output_token_cost": 10 / 1_000_000,
            "max_tokens": 4096,
        },
    }

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
