"""Donchian-channel trend following (Turtle rules; Moskowitz et al. 2012).

Position-trade method on daily bars:

- LONG when today's close exceeds the prior 55-day high (channel breakout),
  protective stop at max(prior 20-day low, close - 2.5*ATR).
- SHORT mirror below the prior 55-day low.
- Exit/trail: opposite 20-day channel or the ATR trail, whichever is nearer.
- Target left open (trend following rides winners); a 2R reference target is
  reported so the decision helper can score reward:risk.
"""


import pandas as pd

from .. import config
from ..indicators import atr, hurst_exponent, rsi
from .base import DataRequirements, TradeSetup, TradingMethod

KEY = "donchian_trend"


def analyze(daily: pd.DataFrame, session: pd.DataFrame | None,
            context: dict) -> TradeSetup:
    closes = daily["Close"].dropna()
    h = hurst_exponent(closes.iloc[-config.HURST_WINDOW:], config.HURST_MAX_LAG)
    atr_daily = float(atr(daily).iloc[-1])
    price = float(closes.iloc[-1])
    cur_rsi = float(rsi(closes).iloc[-1])
    vol20 = daily["Volume"].iloc[-21:-1].mean()
    rel_volume = float(daily["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

    n_e, n_x = config.DONCHIAN_ENTRY, config.DONCHIAN_EXIT
    # Prior channels exclude today's bar (no lookahead).
    hi55 = float(daily["High"].iloc[-(n_e + 1):-1].max())
    lo55 = float(daily["Low"].iloc[-(n_e + 1):-1].min())
    hi20 = float(daily["High"].iloc[-(n_x + 1):-1].max())
    lo20 = float(daily["Low"].iloc[-(n_x + 1):-1].min())

    setup = TradeSetup(
        ticker=context.get("ticker", ""), method_key=KEY, signal="NO_ENTRY",
        playbook="DONCHIAN_BREAKOUT",
        regime="TRENDING" if h >= config.HURST_TREND else
               ("MEAN_REVERTING" if h <= config.HURST_REVERT else "RANDOM_WALK"),
        hurst=h, price=price, atr_daily=atr_daily, rel_volume=rel_volume,
        rsi=cur_rsi, timeframe="position",
        holding_period_hint="weeks to months, ride the trend",
        extras={"hi55": hi55, "lo55": lo55, "hi20": hi20, "lo20": lo20},
    )

    if len(daily) < n_e + 5:
        setup.reasons.append(f"insufficient history: need {n_e + 5} daily bars")
        return setup

    if price > hi55:
        stop = max(lo20, price - config.DONCHIAN_ATR_TRAIL * atr_daily)
        risk = price - stop
        setup.signal, setup.entry, setup.stop = "LONG", price, stop
        setup.target = price + config.RISK_REWARD * risk   # reference only
        setup.reasons.append(
            f"55d breakout long: close {price:.2f} above prior 55d high {hi55:.2f}; "
            f"trail = max(20d low {lo20:.2f}, close - {config.DONCHIAN_ATR_TRAIL}*ATR)"
        )
    elif price < lo55:
        stop = min(hi20, price + config.DONCHIAN_ATR_TRAIL * atr_daily)
        risk = stop - price
        setup.signal, setup.entry, setup.stop = "SHORT", price, stop
        setup.target = price - config.RISK_REWARD * risk
        setup.reasons.append(
            f"55d breakdown short: close {price:.2f} below prior 55d low {lo55:.2f}; "
            f"trail = min(20d high {hi20:.2f}, close + {config.DONCHIAN_ATR_TRAIL}*ATR)"
        )
    else:
        setup.reasons.append(
            f"inside channel: close {price:.2f} within prior 55d range "
            f"[{lo55:.2f}, {hi55:.2f}] — no breakout"
        )
    return setup


from . import register  # noqa: E402

METHOD = register(TradingMethod(
    key=KEY,
    name="Donchian Channel Trend Following",
    paper="Faith, 'The Original Turtle Trading Rules' (2003); Moskowitz, Ooi & "
          "Pedersen, 'Time Series Momentum', JFE 104 (2012)",
    paper_url="https://www.sciencedirect.com/science/article/pii/S0304405X11002613",
    regime_applicability="TRENDING — long-horizon momentum persistence",
    timeframe="position",
    data=DataRequirements(daily_period="2y", needs_intraday=False),
    description=(
        "Classic 55-day Donchian channel breakout with a 20-day opposite-channel "
        "exit and a 2.5x ATR trailing stop. Captures long-horizon time-series "
        "momentum; expects to lose small often and win big rarely."
    ),
    analyze=analyze,
))
