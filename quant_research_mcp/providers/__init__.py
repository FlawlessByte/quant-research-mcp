"""Provider registry. Select with env QUANT_DATA_PROVIDER (default yfinance).

To add a real-time provider (Alpaca, Polygon, IBKR):
    1. Create providers/<name>_provider.py implementing the DataProvider
       protocol (see base.py), reading its API key from the environment.
    2. Register a factory below.
    3. Run with QUANT_DATA_PROVIDER=<name>.
"""

import os

from .base import DataProvider
from .yfinance_provider import YFinanceProvider

_FACTORIES = {
    "yfinance": YFinanceProvider,
}

_active: DataProvider | None = None


def get_provider() -> DataProvider:
    global _active
    if _active is None:
        name = os.environ.get("QUANT_DATA_PROVIDER", "yfinance").lower()
        factory = _FACTORIES.get(name)
        if factory is None:
            valid = ", ".join(sorted(_FACTORIES))
            raise ValueError(f"unknown QUANT_DATA_PROVIDER '{name}'. Valid: {valid}")
        _active = factory()
    return _active


__all__ = ["DataProvider", "get_provider"]
