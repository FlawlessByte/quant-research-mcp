"""yfinance-backed provider (default). Free, ~15-minute delayed."""

from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from ..cache import TTL_DAILY, TTL_EVENTS, TTL_INTRADAY, TTL_NEWS, TTL_SECTOR, ttl_cache


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


class YFinanceProvider:
    name = "yfinance"

    @ttl_cache(TTL_DAILY)
    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        return _flatten(df).dropna(how="all")

    def fetch_daily_batch(self, tickers: list[str], period: str
                          ) -> dict[str, pd.DataFrame]:
        return self._fetch_daily_batch(tuple(tickers), period)

    @ttl_cache(TTL_DAILY)
    def _fetch_daily_batch(self, tickers: tuple[str, ...], period: str
                           ) -> dict[str, pd.DataFrame]:
        raw = yf.download(list(tickers), period=period, interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                df = (raw[t].dropna(how="all")
                      if isinstance(raw.columns, pd.MultiIndex) else raw)
                if not df.empty and df["Close"].notna().sum() >= 30:
                    out[t] = df
            except KeyError:
                continue
        return out

    @ttl_cache(TTL_INTRADAY)
    def fetch_intraday(self, ticker: str, interval: str, period: str
                       ) -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        return _flatten(df).dropna(how="all")

    @ttl_cache(TTL_NEWS)
    def get_headlines(self, ticker: str, limit: int) -> list[dict]:
        try:
            items = yf.Ticker(ticker).news or []
        except Exception:
            return []
        out = []
        for item in items[:limit]:
            c = item.get("content") or {}
            title = c.get("title")
            if not title:
                continue
            out.append({
                "title": title,
                "summary": (c.get("summary") or "")[:280],
                "published": c.get("pubDate", ""),
                "provider": (c.get("provider") or {}).get("displayName", ""),
            })
        return out

    @ttl_cache(TTL_EVENTS)
    def get_events(self, ticker: str) -> dict:
        out = {"next_earnings": None, "days_to_earnings": None,
               "recent_earnings": [], "ex_dividend": None}
        today = datetime.now(UTC).date()
        try:
            tk = yf.Ticker(ticker)
            # Primary source: the calendar dict (no lxml dependency).
            cal = getattr(tk, "calendar", None) or {}
            if isinstance(cal, dict):
                edates = cal.get("Earnings Date") or []
                future = sorted(d for d in edates if d >= today)
                if future:
                    out["next_earnings"] = future[0].isoformat()
                    out["days_to_earnings"] = (future[0] - today).days
                exdiv = cal.get("Ex-Dividend Date")
                if exdiv is not None:
                    out["ex_dividend"] = str(exdiv)
            # Optional history (requires lxml; ignore if unavailable).
            try:
                ed = tk.earnings_dates
                if ed is not None and not ed.empty:
                    past = sorted(d for d in ed.index if d.date() <= today)
                    out["recent_earnings"] = [d.date().isoformat() for d in past[-4:]]
                    if out["next_earnings"] is None:
                        fut = sorted(d for d in ed.index if d.date() > today)
                        if fut:
                            out["next_earnings"] = fut[0].date().isoformat()
                            out["days_to_earnings"] = (fut[0].date() - today).days
            except Exception:
                pass
        except Exception:
            pass
        return out

    def get_quote(self, ticker: str) -> dict | None:
        return None  # yfinance has no true real-time quote

    @ttl_cache(TTL_SECTOR)
    def get_sector(self, ticker: str) -> str | None:
        try:
            return yf.Ticker(ticker).info.get("sector")
        except Exception:
            return None
