import asyncio
from unittest import mock

import pytest

from concurrent_openai.rate_limiter import RateLimiter


@pytest.mark.asyncio
@mock.patch(
    "concurrent_openai.rate_limiter.settings.OPENAI_MODEL_DETAILS",
    {
        "gpt-4": {
            "rpm": 120,
            "tkm": 600,
            "input_token_cost": 1,
            "output_token_cost": 3,
        }
    },
)
async def test_rate_limiter_refill_mechanism():
    rate_limiter = RateLimiter("gpt-4")
    # Set the refill interval to a very low value to speed up the test (normally this should always be 1)
    with mock.patch.object(rate_limiter, "refill_interval", 0.01):
        assert rate_limiter.rps == 2  # 120 / 60
        assert rate_limiter.tks == 10  # 600 / 60

        # The buckets must be initialized with rps and tks
        assert rate_limiter.request_bucket.available_amount == 2
        assert rate_limiter.token_bucket.available_amount == 10

        # Simulate a request that consumes 10 tokens
        await rate_limiter.acquire(10)
        assert rate_limiter.request_bucket.available_amount == 1
        assert rate_limiter.token_bucket.available_amount == 0

        # Buckets must be refilled after the refill interval
        await asyncio.sleep(0.0105)
        assert rate_limiter.request_bucket.available_amount == 3
        assert rate_limiter.token_bucket.available_amount == 10

        # Buckets are continuously refilled
        await asyncio.sleep(0.0105)
        assert rate_limiter.request_bucket.available_amount == 5
        assert rate_limiter.token_bucket.available_amount == 20

    await rate_limiter.cleanup()


@pytest.mark.asyncio
@mock.patch(
    "concurrent_openai.rate_limiter.settings.OPENAI_MODEL_DETAILS",
    {
        "gpt-4": {
            "rpm": 120,
            "tkm": 6000,
            "input_token_cost": 1,
            "output_token_cost": 3,
        }
    },
)
async def test_rate_limiter_throttling():
    rate_limiter = RateLimiter("gpt-4")
    with mock.patch.object(rate_limiter, "refill_interval", 0.02):
        assert rate_limiter.rps == 2  # 120 / 60
        assert rate_limiter.tks == 100

        loop = asyncio.get_event_loop()
        start_time = loop.time()
        await asyncio.gather(
            rate_limiter.acquire(10),
            rate_limiter.acquire(10),
            rate_limiter.acquire(10),
            rate_limiter.acquire(10),
        )
        end_time = loop.time()

        elapsed_time = end_time - start_time

        # The requests should be throttled to 2 per refill interval
        assert 0.046 >= elapsed_time >= 0.04

        # 2 requests initially, 4 consumed, buckets refilled 2 times, thus 2 requests available
        assert rate_limiter.request_bucket.available_amount == 2

        # 100 tokens initially, 4*10 consumed, buckets refilled 2 times, thus 100 - 4 * 10 + 2 * 100
        assert rate_limiter.token_bucket.available_amount == 260

    await rate_limiter.cleanup()
