"""Method registry.

`_REGISTRY` maps method key -> TradingMethod. Each method module calls
`register(...)` at import time; importing this package imports every bundled
method so the registry is populated on first use.

To add a future paper:
    1. Create quant_research_mcp/methods/<your_paper>.py
    2. Build a TradingMethod and call register(it) at module top level.
    3. Import it below (one line). Nothing else changes.
"""

from .base import TradeSetup, TradingMethod

_REGISTRY: dict[str, TradingMethod] = {}


def register(method: TradingMethod) -> TradingMethod:
    if method.key in _REGISTRY:
        raise ValueError(f"method key already registered: {method.key}")
    _REGISTRY[method.key] = method
    return method


def get(key: str) -> TradingMethod:
    if key not in _REGISTRY:
        raise KeyError(key)
    return _REGISTRY[key]


def has(key: str) -> bool:
    return key in _REGISTRY


def list_methods() -> list[TradingMethod]:
    return list(_REGISTRY.values())


# --- Bundled methods (import to register). Add future papers here. ---------
from . import (
    donchian_trend,  # noqa: E402,F401
    hurst_regime,  # noqa: E402,F401
    pairs_cointegration,  # noqa: E402,F401
    rsi2_reversion,  # noqa: E402,F401
    xs_momentum,  # noqa: E402,F401
)

__all__ = ["TradingMethod", "TradeSetup", "register", "get", "has", "list_methods"]
