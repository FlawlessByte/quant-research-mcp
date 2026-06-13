"""In-memory TTL cache for data-layer calls.

Keeps repeated tool calls from re-downloading the same history within a
session and softens rate limits. Process-local; entries expire on a
monotonic clock. Not persisted — restarting the server clears it.
"""

import functools
import time
from threading import Lock

_STORE: dict = {}
_LOCK = Lock()

# Default TTLs (seconds) per data class.
TTL_DAILY = 15 * 60
TTL_INTRADAY = 60
TTL_NEWS = 5 * 60
TTL_EVENTS = 60 * 60
TTL_SECTOR = 60 * 60


def ttl_cache(ttl: float):
    """Memoize a function for `ttl` seconds, keyed on its args."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__module__, fn.__qualname__, args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with _LOCK:
                hit = _STORE.get(key)
                if hit is not None and now - hit[0] < ttl:
                    return hit[1]
            value = fn(*args, **kwargs)
            with _LOCK:
                _STORE[key] = (now, value)
            return value
        wrapper.cache_clear = lambda: _clear_prefix(fn.__qualname__)  # type: ignore[attr-defined]
        return wrapper
    return decorator


def _clear_prefix(qualname: str) -> None:
    with _LOCK:
        for k in [k for k in _STORE if k[1] == qualname]:
            del _STORE[k]


def clear_all() -> None:
    with _LOCK:
        _STORE.clear()
