"""Market data access. Delegates to the active provider (see providers/).

Import surface kept stable: screener, methods and the server import from
here and never touch a concrete provider.
"""

import pandas as pd

from . import config
from .providers import get_provider


def fetch_daily_batch(tickers: list[str], period: str = config.DAILY_LOOKBACK
                      ) -> dict[str, pd.DataFrame]:
    return get_provider().fetch_daily_batch(tickers, period)


def fetch_daily(ticker: str, period: str = config.DAILY_LOOKBACK) -> pd.DataFrame:
    return get_provider().fetch_daily(ticker, period)


def fetch_intraday(ticker: str, interval: str = config.INTRADAY_INTERVAL,
                   period: str = config.INTRADAY_PERIOD) -> pd.DataFrame:
    return get_provider().fetch_intraday(ticker, interval, period)


def get_headlines(ticker: str, limit: int = 8) -> list[dict]:
    return get_provider().get_headlines(ticker, limit)


def get_events(ticker: str) -> dict:
    return get_provider().get_events(ticker)


def get_sector(ticker: str) -> str | None:
    return get_provider().get_sector(ticker)


def last_session(intraday: pd.DataFrame) -> pd.DataFrame:
    """Bars belonging to the most recent trading session."""
    if intraday.empty:
        return intraday
    last_day = intraday.index[-1].date()
    return intraday[[ts.date() == last_day for ts in intraday.index]]


def prior_sessions(intraday: pd.DataFrame) -> pd.DataFrame:
    """Bars from every session except the most recent."""
    if intraday.empty:
        return intraday
    last_day = intraday.index[-1].date()
    return intraday[[ts.date() != last_day for ts in intraday.index]]
