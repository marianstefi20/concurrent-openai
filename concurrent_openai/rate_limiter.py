import asyncio

import structlog

from .settings import settings

LOGGER = structlog.get_logger(__name__)


class Bucket:
    def __init__(self, amount: int):
        self.available_amount = amount
        self.cond = asyncio.Condition()

    async def acquire(self, amount: int = 1):
        async with self.cond:
            while self.available_amount < amount:
                await self.cond.wait()
            self.available_amount -= amount

    async def release(self, amount: int = 1):
        async with self.cond:
            self.available_amount += amount
            self.cond.notify_all()


class RateLimiter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name not in settings.OPENAI_MODEL_DETAILS:
            raise ValueError(f"Unsupported model: {model_name}")

        model_limits = settings.OPENAI_MODEL_DETAILS[model_name]

        self.rps = int(model_limits["rpm"] / 60)
        self.tks = int(model_limits["tkm"] / 60)
        self.tokens_estimation = 0

        self.request_bucket = Bucket(self.rps)
        self.token_bucket = Bucket(self.tks)

        self.refill_request_task = asyncio.create_task(self._refill_request_limiter())
        self.refill_token_task = asyncio.create_task(self._refill_token_bucket())

        self.loop = asyncio.get_event_loop()
        self.last_request_time = self.loop.time()
        self.refill_interval: float = (
            1  # Refill rate is 1 token per second, but for tests we can set it to 0.01
        )
        self.request_slot_lock = asyncio.Lock()

    async def acquire(self, tokens_estimation: int):
        await self.wait_for_next_request_slot()
        self.tokens_estimation = tokens_estimation
        await self.token_bucket.acquire(self.tokens_estimation)
        await self.request_bucket.acquire(1)

    async def wait_for_next_request_slot(self):
        async with self.request_slot_lock:
            elapsed_since_last_request = self.loop.time() - self.last_request_time

            ideal_interval = self.refill_interval / self.rps
            if elapsed_since_last_request < ideal_interval:
                await asyncio.sleep(ideal_interval - elapsed_since_last_request)

            self.last_request_time = self.loop.time()

    async def release(self, tokens: int):
        # Because tokens estimation is not accurate, fine tune the token bucket
        if self.tokens_estimation < tokens:
            LOGGER.warning(
                "Underestimated the number of tokens required for this request",
                tokens_estimation=self.tokens_estimation,
                tokens=tokens,
            )
            await self.token_bucket.acquire(tokens - self.tokens_estimation)
        elif self.tokens_estimation > tokens:
            await self.token_bucket.release(self.tokens_estimation - tokens)

    async def cleanup(self):
        self.refill_request_task.cancel()
        self.refill_token_task.cancel()

        try:
            await self.refill_request_task
        except asyncio.CancelledError:
            pass

        try:
            await self.refill_token_task
        except asyncio.CancelledError:
            pass

    async def _refill_request_limiter(self):
        while True:
            await asyncio.sleep(self.refill_interval)
            await self.request_bucket.release(self.rps)

    async def _refill_token_bucket(self):
        while True:
            await asyncio.sleep(self.refill_interval)
            await self.token_bucket.release(self.tks)
