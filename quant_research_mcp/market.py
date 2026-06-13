"""Market context: index tape, VIX, sector rotation. One batched download."""

import pandas as pd

from . import data
from .indicators import ema

INDEXES = ["SPY", "QQQ", "IWM"]
VIX = "^VIX"
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLI": "Industrials", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services",
}


def _snapshot(df: pd.DataFrame) -> dict:
    closes = df["Close"].dropna()
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    e20 = float(ema(closes, 20).iloc[-1])
    return {
        "last": round(last, 2),
        "day_pct": round((last / prev - 1) * 100, 2),
        "above_20d_ema": last > e20,
    }


def context() -> dict:
    tickers = INDEXES + [VIX] + list(SECTORS)
    daily = data.fetch_daily_batch(tickers, "3mo")

    out: dict = {"indexes": {}, "sectors": [], "vix": None}
    for t in INDEXES:
        if t in daily:
            out["indexes"][t] = _snapshot(daily[t])
    if VIX in daily:
        closes = daily[VIX]["Close"].dropna()
        out["vix"] = {"last": round(float(closes.iloc[-1]), 2),
                      "day_change": round(float(closes.iloc[-1] - closes.iloc[-2]), 2)}

    sect = []
    for etf, name in SECTORS.items():
        if etf in daily:
            snap = _snapshot(daily[etf])
            sect.append({"etf": etf, "sector": name, **snap})
    sect.sort(key=lambda s: s["day_pct"], reverse=True)
    out["sectors"] = sect
    advancers = sum(1 for s in sect if s["day_pct"] > 0)
    out["breadth"] = {"sectors_advancing": advancers,
                      "sectors_total": len(sect),
                      "risk_on": advancers > len(sect) / 2}
    return out
