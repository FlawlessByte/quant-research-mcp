"""Data-provider contract.

A provider supplies market data to the rest of the package. The default is
yfinance (free, ~15-min delayed). Real-time providers (Alpaca, Polygon, IBKR)
implement the same protocol and register under a name; select with the
QUANT_DATA_PROVIDER environment variable. Tools and methods never import a
concrete provider — they go through quant_research_mcp.data.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataProvider(Protocol):
    """All frames use columns Open/High/Low/Close/Volume, DatetimeIndex."""

    name: str

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame: ...

    def fetch_daily_batch(self, tickers: list[str], period: str
                          ) -> dict[str, pd.DataFrame]: ...

    def fetch_intraday(self, ticker: str, interval: str, period: str
                       ) -> pd.DataFrame: ...

    def get_headlines(self, ticker: str, limit: int) -> list[dict]: ...

    def get_events(self, ticker: str) -> dict:
        """Earnings/dividend calendar.

        Returns {next_earnings: iso str|None, days_to_earnings: int|None,
                 recent_earnings: [iso str], ex_dividend: iso str|None}.
        """
        ...

    def get_quote(self, ticker: str) -> dict | None:
        """Real-time quote {price, bid, ask, ts} where supported, else None."""
        ...

    def get_sector(self, ticker: str) -> str | None: ...
