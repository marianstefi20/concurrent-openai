import asyncio
import time

import pytest

from concurrent_openai.rate_limiter import RateLimiter


def truncate(value: float, decimals: int = 3) -> float:
    return int(value * 10**decimals) / 10**decimals


@pytest.mark.asyncio
async def test_basic_refill():
    """
    Test that the rate limiter refills tokens over time when capacity is not exceeded.
    Make a request that consumes fewer tokens than capacity, wait,
    and check that tokens are refilled properly.
    """
    capacity = 10
    fill_rate = 2  # tokens per second
    limiter = RateLimiter(capacity=capacity, fill_rate=fill_rate)

    # Initially, tokens == capacity
    assert limiter.tokens == capacity

    # Acquire 5 tokens
    await limiter.acquire(5)
    # New tokens should be capacity - 5 = 5
    assert limiter.tokens == 5

    # Wait 1 second so we expect fill_rate more tokens
    await asyncio.sleep(1.0)
    # Acquire 1 token to trigger lazy refill
    await limiter.acquire(1)

    assert 6 == pytest.approx(limiter.tokens, abs=0.01)


@pytest.mark.asyncio
async def test_burst_capacity():
    """
    Test that multiple requests up to 'capacity' can proceed immediately,
    and the next request is delayed until tokens are replenished.
    """
    capacity = 3
    fill_rate = 1  # 1 token/second
    limiter = RateLimiter(capacity=capacity, fill_rate=fill_rate)

    # Acquire up to capacity immediately
    start_time = time.monotonic()
    # Acquire should happen with no delay for these 3:
    await limiter.acquire(1)
    await limiter.acquire(1)
    await limiter.acquire(1)
    immediate_elapsed = time.monotonic() - start_time

    # We expect nearly no wait for these calls (all within capacity).
    # We'll allow a small scheduling margin.
    assert (
        immediate_elapsed < 0.001
    ), f"Expected near-instant acquisition, got {immediate_elapsed:.3f}s"

    # Next acquire (the 4th token) should force a wait of ~1s (since fill_rate=1)
    start_time = time.monotonic()
    await limiter.acquire(1)
    forced_wait_elapsed = time.monotonic() - start_time

    # Should be ~1 second (give or take scheduling delays).
    # We'll check for at least 0.9 to confirm we waited.
    assert forced_wait_elapsed >= 0.9, f"Expected about 1s wait, got {forced_wait_elapsed:.3f}s"


@pytest.mark.asyncio
async def test_with_minimum_spacing():
    """
    Test that the rate limiter enforces the minimum spacing between calls,
    even if tokens are still available.
    """
    capacity = 5
    fill_rate = 5  # 5 tokens/sec
    min_spacing = 0.5  # half a second
    limiter = RateLimiter(capacity=capacity, fill_rate=fill_rate, minimum_spacing=min_spacing)

    # First call should be immediate
    start_time = time.monotonic()
    await limiter.acquire(1)
    first_elapsed = time.monotonic() - start_time
    assert first_elapsed < 0.001, f"Expected near-instant first call, got {first_elapsed:.3f}s"

    # Second call should be forced to wait ~0.5s due to min_spacing
    start_time = time.monotonic()
    await limiter.acquire(1)
    second_elapsed = time.monotonic() - start_time
    assert second_elapsed >= 0.4, f"Expected ~0.5s spacing, got {second_elapsed:.3f}s"


@pytest.mark.asyncio
async def test_no_minimum_spacing():
    """
    If minimum_spacing = 0, then calls that have enough tokens should proceed
    immediately without extra delay.
    """
    capacity = 5
    fill_rate = 10  # tokens/sec, quite high to avoid forced waits from capacity
    limiter = RateLimiter(capacity=capacity, fill_rate=fill_rate, minimum_spacing=0.0)

    # Make two quick requests in succession
    start_time = time.monotonic()
    await limiter.acquire(1)
    mid_time = time.monotonic()
    await limiter.acquire(1)
    total_elapsed = time.monotonic() - start_time
    first_elapsed = mid_time - start_time
    second_elapsed = total_elapsed - first_elapsed

    # We expect minimal or no delay for both acquires since we have plenty of tokens.
    assert first_elapsed < 0.1, f"Expected near-instant first call, got {first_elapsed:.3f}s"
    assert second_elapsed < 0.1, f"Expected near-instant second call, got {second_elapsed:.3f}s"


@pytest.mark.asyncio
async def test_parallel_acquisitions():
    """
    Test concurrent behavior of the rate limiter:
    1. First task acquires immediately (2 tokens)
    2. Second task waits ~1s (for 2 tokens to refill)
    3. Third task waits ~2s (for 2 more tokens to refill)
    """
    capacity = 2
    fill_rate = 2  # tokens per second
    limiter = RateLimiter(capacity=capacity, fill_rate=fill_rate)

    async def do_acquire(idx, tokens):
        start = time.monotonic()
        await limiter.acquire(tokens)
        duration = time.monotonic() - start
        return idx, duration

    # Launch 3 concurrent tasks, each requesting 2 tokens
    tasks = [asyncio.create_task(do_acquire(i, 2)) for i in range(3)]
    results = await asyncio.gather(*tasks)
    results_sorted = sorted(results, key=lambda x: x[1])

    # First task (idx=0) should complete immediately
    assert results_sorted[0][0] == 0
    assert results_sorted[0][1] == pytest.approx(0, abs=0.01)

    # Second task should wait ~1 second
    assert results_sorted[1][1] == pytest.approx(1.0, abs=0.01)

    # Third task should wait ~2 seconds
    assert results_sorted[2][1] == pytest.approx(2.0, abs=0.01)
