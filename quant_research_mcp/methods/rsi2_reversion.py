"""RSI(2) mean reversion (Connors & Alvarez 2009).

Swing method on daily bars:

- LONG when RSI(2) < 10 AND close > 200d SMA (pullback in an uptrend).
- SHORT when RSI(2) > 90 AND close < 200d SMA.
- Exit when close crosses the 5d SMA (reported as target proxy);
  protective stop 1.5x ATR beyond entry.
- Holds 2-7 trading days typically.
"""


import pandas as pd

from .. import config
from ..indicators import atr, hurst_exponent, rsi, sma
from .base import DataRequirements, TradeSetup, TradingMethod

KEY = "rsi2_reversion"


def analyze(daily: pd.DataFrame, session: pd.DataFrame | None,
            context: dict) -> TradeSetup:
    closes = daily["Close"].dropna()
    h = hurst_exponent(closes.iloc[-config.HURST_WINDOW:], config.HURST_MAX_LAG)
    atr_daily = float(atr(daily).iloc[-1])
    price = float(closes.iloc[-1])
    rsi2 = float(rsi(closes, period=2).iloc[-1])
    rsi14 = float(rsi(closes).iloc[-1])
    ma200 = float(sma(closes, config.RSI2_TREND_MA).iloc[-1])
    ma5 = float(sma(closes, config.RSI2_EXIT_MA).iloc[-1])
    vol20 = daily["Volume"].iloc[-21:-1].mean()
    rel_volume = float(daily["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

    setup = TradeSetup(
        ticker=context.get("ticker", ""), method_key=KEY, signal="NO_ENTRY",
        playbook="RSI2_PULLBACK",
        regime="MEAN_REVERTING" if h <= config.HURST_REVERT else
               ("TRENDING" if h >= config.HURST_TREND else "RANDOM_WALK"),
        hurst=h, price=price, atr_daily=atr_daily, rel_volume=rel_volume,
        rsi=rsi14, timeframe="swing",
        holding_period_hint="2-7 trading days, exit at 5d SMA",
        extras={"rsi2": rsi2, "sma200": ma200, "sma5": ma5},
    )

    if len(closes) < config.RSI2_TREND_MA + 5:
        setup.reasons.append(
            f"insufficient history: need {config.RSI2_TREND_MA + 5} daily bars")
        return setup

    if rsi2 < config.RSI2_ENTRY and price > ma200:
        stop = price - 1.5 * atr_daily
        setup.signal, setup.entry, setup.stop = "LONG", price, stop
        # Exit is the 5d SMA cross; report it as the target proxy (>= entry
        # when the snapback works). Use max so R:R is at least defined.
        setup.target = max(ma5, price + (price - stop))
        setup.reasons.append(
            f"RSI(2) {rsi2:.1f} < {config.RSI2_ENTRY:.0f} oversold with close "
            f"{price:.2f} above 200d SMA {ma200:.2f} — pullback long; exit on "
            f"close above 5d SMA {ma5:.2f}"
        )
    elif rsi2 > config.RSI2_ENTRY_SHORT and price < ma200:
        stop = price + 1.5 * atr_daily
        setup.signal, setup.entry, setup.stop = "SHORT", price, stop
        setup.target = min(ma5, price - (stop - price))
        setup.reasons.append(
            f"RSI(2) {rsi2:.1f} > {config.RSI2_ENTRY_SHORT:.0f} overbought with "
            f"close {price:.2f} below 200d SMA {ma200:.2f} — rally short; exit "
            f"on close below 5d SMA {ma5:.2f}"
        )
    else:
        side = "above" if price > ma200 else "below"
        setup.reasons.append(
            f"no extreme: RSI(2) {rsi2:.1f} (need <{config.RSI2_ENTRY:.0f} long "
            f"above 200d SMA or >{config.RSI2_ENTRY_SHORT:.0f} short below); "
            f"price {side} 200d SMA {ma200:.2f}"
        )
    return setup


from . import register  # noqa: E402

METHOD = register(TradingMethod(
    key=KEY,
    name="RSI(2) Mean Reversion",
    paper="Connors & Alvarez, 'Short Term Trading Strategies That Work' (2009)",
    paper_url="https://www.quantifiedstrategies.com/rsi2-trading-strategy/",
    regime_applicability="MEAN_REVERTING pullbacks within a long-term uptrend",
    timeframe="swing",
    data=DataRequirements(daily_period="2y", needs_intraday=False),
    description=(
        "Buys 2-period RSI capitulation (<10) when price holds above the 200-day "
        "SMA, exits on the close crossing the 5-day SMA; mirror short below the "
        "200-day. High win rate, short holds, depends on the long-term trend "
        "filter to avoid catching knives."
    ),
    analyze=analyze,
))
