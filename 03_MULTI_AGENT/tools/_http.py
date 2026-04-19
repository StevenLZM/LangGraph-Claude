"""HTTP 工具公共依赖：复用 httpx.AsyncClient + tenacity 重试。"""
from __future__ import annotations

import logging

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def make_client(timeout: float = 30.0, **kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        **kwargs,
    )


def retrying() -> AsyncRetrying:
    """统一重试策略：指数退避，最多 3 次，仅对 HTTP/网络错误重试。"""
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )


async def safe_get_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict | list | None:
    try:
        async for attempt in retrying():
            with attempt:
                resp = await client.get(url, **kwargs)
                resp.raise_for_status()
                return resp.json()
    except (httpx.HTTPError, RetryError) as e:
        logger.warning("[http] GET %s failed: %s", url, e)
        return None


async def safe_post_json(client: httpx.AsyncClient, url: str, **kwargs) -> dict | list | None:
    try:
        async for attempt in retrying():
            with attempt:
                resp = await client.post(url, **kwargs)
                resp.raise_for_status()
                return resp.json()
    except (httpx.HTTPError, RetryError) as e:
        logger.warning("[http] POST %s failed: %s", url, e)
        return None


async def safe_get_text(client: httpx.AsyncClient, url: str, **kwargs) -> str | None:
    try:
        async for attempt in retrying():
            with attempt:
                resp = await client.get(url, **kwargs)
                resp.raise_for_status()
                return resp.text
    except (httpx.HTTPError, RetryError) as e:
        logger.warning("[http] GET-text %s failed: %s", url, e)
        return None
