import asyncio
import time
from typing import Optional

import structlog

LOGGER = structlog.get_logger(__name__)


class RateLimiter:
    """A token bucket rate limiter implementation.

    This rate limiter uses the token bucket algorithm to control the rate of actions.
    It supports both steady-state rate limiting and optional minimum spacing between requests.

    Attributes:
        capacity: Maximum number of tokens that can accumulate (burst limit)
        fill_rate: Number of tokens added per second (steady-state rate)
        minimum_spacing: Minimal time in seconds between requests
    """

    def __init__(
        self,
        capacity: float,
        fill_rate: float,
        minimum_spacing: float = 0.0,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            capacity: Maximum number of tokens that can accumulate (burst limit)
            fill_rate: Number of tokens added per second (steady-state rate)
            minimum_spacing: Minimal time in seconds between requests.
                           Set to 0.0 (default) to allow bursting up to capacity.

        Raises:
            ValueError: If capacity, fill_rate or minimum_spacing are negative
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if fill_rate <= 0:
            raise ValueError("Fill rate must be positive")
        if minimum_spacing < 0:
            raise ValueError("Minimum spacing cannot be negative")

        self._capacity = capacity
        self._tokens = capacity
        self._fill_rate = fill_rate
        self._minimum_spacing = minimum_spacing

        self._last_refill_time = time.monotonic()
        self._last_request_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> float:
        """Maximum number of tokens that can accumulate."""
        return self._capacity

    @property
    def tokens(self) -> float:
        """Current number of available tokens."""
        return self._tokens

    @property
    def fill_rate(self) -> float:
        """Number of tokens added per second."""
        return self._fill_rate

    @property
    def minimum_spacing(self) -> float:
        """Minimum time required between requests."""
        return self._minimum_spacing

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            raise ValueError("Number of tokens must be positive")
        if tokens > self.capacity:
            raise ValueError("Requested tokens cannot exceed the bucket capacity")

        while True:
            async with self._lock:
                now = time.monotonic()
                wait_time = self._calculate_wait_time(now, tokens)

                if wait_time <= 0:
                    self._tokens -= tokens
                    self._last_request_time = now
                    return

            await asyncio.sleep(wait_time)

    def _calculate_wait_time(self, now: float, requested_tokens: float) -> float:
        wait_time = 0.0

        # Check minimum spacing requirement
        if self._minimum_spacing > 0 and self._last_request_time is not None:
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < self._minimum_spacing:
                wait_time = self._minimum_spacing - elapsed_since_last

        # Refill tokens
        self._refill(now)

        # Check if we need to wait for token refill
        if self._tokens < requested_tokens:
            missing = requested_tokens - self._tokens
            refill_wait = missing / self._fill_rate
            wait_time = max(wait_time, refill_wait)

        return wait_time

    def _refill(self, now: float) -> None:
        """Refill the token bucket based on elapsed time.

        Args:
            now: Current timestamp
        """
        elapsed = now - self._last_refill_time
        if elapsed > 0:
            added_tokens = elapsed * self._fill_rate
            self._tokens = min(self._capacity, self._tokens + added_tokens)
            self._last_refill_time = now

            LOGGER.debug(
                "Tokens refilled",
                added=added_tokens,
                current=self._tokens,
                capacity=self._capacity,
                elapsed=elapsed,
            )

    def __repr__(self) -> str:
        """Return string representation of the rate limiter."""
        return (
            f"RateLimiter(capacity={self._capacity}, "
            f"fill_rate={self._fill_rate}, "
            f"minimum_spacing={self._minimum_spacing}, "
            f"current_tokens={self._tokens:.2f})"
        )
