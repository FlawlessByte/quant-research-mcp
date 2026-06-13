"""Pre-trade screener: rank a universe by gap, relative volume and range.

Score favours names that are moving today (gap), trading above normal volume
(participation) and have enough daily range (ATR%) to pay for the trade. Hard
filters drop illiquid or cheap names.
"""

from dataclasses import dataclass

import pandas as pd

from . import config
from .indicators import atr


@dataclass
class ScreenResult:
    ticker: str
    price: float
    gap_pct: float
    rel_volume: float
    atr_pct: float
    dollar_volume: float
    score: float

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "price": round(self.price, 2),
            "gap_pct": round(self.gap_pct, 2),
            "rel_volume": round(self.rel_volume, 2),
            "atr_pct": round(self.atr_pct, 2),
            "avg_dollar_volume_m": round(self.dollar_volume / 1e6, 1),
            "score": round(self.score, 3),
        }


def screen(daily: dict[str, pd.DataFrame], top_n: int = config.TOP_N) -> list[ScreenResult]:
    results: list[ScreenResult] = []
    for ticker, df in daily.items():
        if len(df) < 30:
            continue
        close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        today_open = df["Open"].iloc[-1]
        vol20 = df["Volume"].iloc[-21:-1].mean()
        if vol20 <= 0 or prev_close <= 0:
            continue

        price = float(close)
        dollar_volume = float(vol20 * df["Close"].iloc[-21:-1].mean())
        if price < config.MIN_PRICE or dollar_volume < config.MIN_DOLLAR_VOLUME:
            continue

        gap_pct = float((today_open / prev_close - 1) * 100)
        rel_volume = float(df["Volume"].iloc[-1] / vol20)
        atr_pct = float(atr(df).iloc[-1] / close * 100)

        score = abs(gap_pct) * 2.0 + rel_volume * 1.0 + atr_pct * 0.5
        results.append(
            ScreenResult(ticker, price, gap_pct, rel_volume, atr_pct, dollar_volume, score)
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
