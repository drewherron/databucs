"""Small HTTP helper with retries. Standard library only, so the fetch job needs no extra
dependencies on serverless."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable

USER_AGENT = "Mozilla/5.0 (compatible; databucs)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """A request failed after all retries (or with a non-retryable status)."""


def get_bytes(
    url: str,
    timeout: float = 120.0,
    attempts: int = 4,
    backoff_seconds: float = 2.0,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET `url` and return the body, retrying transient failures with exponential backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener(request, timeout=timeout) as response:  # type: ignore[attr-defined]
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_STATUS:
                raise FetchError(f"HTTP {e.code} for {url}") from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
        if attempt < attempts - 1:
            sleep(backoff_seconds * 2**attempt)
    raise FetchError(f"Failed after {attempts} attempts: {url} ({last_error})") from last_error
