"""Hurst-regime intraday method (arXiv:2205.11122).

The daily Hurst exponent selects the playbook:

- TREND  (H >= 0.55): 15-minute Opening Range Breakout, confirmed by VWAP side
  and 5m EMA9/EMA20 alignment. Stop at the opposite side of the opening range;
  target pays RISK_REWARD times the risk.
- REVERT (H <= 0.45): fade extensions beyond VWAP_FADE_SIGMA from VWAP when 5m
  RSI is at an extreme, targeting a return to VWAP; stop at the session extreme.
- MIXED  (in between): breakout only with elevated relative volume, else NO_ENTRY.
"""

import pandas as pd

from .. import config
from ..indicators import atr, ema, hurst_exponent, rsi, vwap
from .base import TradeSetup

KEY = "hurst_regime_orb"


def _classify(h: float) -> tuple[str, str]:
    if h >= config.HURST_TREND:
        return "TRENDING", "TREND_ORB"
    if h <= config.HURST_REVERT:
        return "MEAN_REVERTING", "REVERT_VWAP_FADE"
    return "RANDOM_WALK", "MIXED"


def analyze(daily: pd.DataFrame, session: pd.DataFrame, context: dict) -> TradeSetup:
    rel_volume = float(context.get("rel_volume", _rel_volume_from_daily(daily)))
    closes = daily["Close"].dropna()
    h = hurst_exponent(closes.iloc[-config.HURST_WINDOW:], config.HURST_MAX_LAG)
    regime, playbook = _classify(h)
    atr_daily = float(atr(daily).iloc[-1])

    n_or = config.OPENING_RANGE_BARS
    or_high = float(session["High"].iloc[:n_or].max())
    or_low = float(session["Low"].iloc[:n_or].min())
    price = float(session["Close"].iloc[-1])
    session_vwap = vwap(session)
    cur_vwap = float(session_vwap.iloc[-1])
    e9 = float(ema(session["Close"], 9).iloc[-1])
    e20 = float(ema(session["Close"], 20).iloc[-1])
    cur_rsi = float(rsi(session["Close"]).iloc[-1])
    dev_sigma = float((session["Close"] - session_vwap).std())

    setup = TradeSetup(
        ticker=context.get("ticker", ""), method_key=KEY, signal="NO_ENTRY",
        playbook=playbook, regime=regime, hurst=h, price=price,
        atr_daily=atr_daily, rel_volume=rel_volume, rsi=cur_rsi,
        timeframe="intraday", holding_period_hint="minutes to hours, flat by close",
        extras={"vwap": cur_vwap, "or_high": or_high, "or_low": or_low,
                "ema9": e9, "ema20": e20},
    )

    if len(session) <= n_or:
        setup.reasons.append("session too young: opening range still forming")
        return setup

    if rel_volume < config.MIN_REL_VOLUME and playbook != "REVERT_VWAP_FADE":
        setup.reasons.append(
            f"relative volume {rel_volume:.2f} below {config.MIN_REL_VOLUME} minimum"
        )
        return setup

    if playbook in ("TREND_ORB", "MIXED"):
        need_extra = playbook == "MIXED"
        vol_ok = rel_volume >= (config.MIN_REL_VOLUME * 1.5 if need_extra
                                else config.MIN_REL_VOLUME)
        long_ok = price > or_high and price > cur_vwap and e9 > e20
        short_ok = price < or_low and price < cur_vwap and e9 < e20
        if vol_ok and long_ok:
            risk = price - or_low
            setup.signal, setup.entry, setup.stop = "LONG", price, or_low
            setup.target = price + config.RISK_REWARD * risk
            setup.reasons.append(
                f"ORB long: price {price:.2f} above OR high {or_high:.2f}, above "
                f"VWAP {cur_vwap:.2f}, EMA9>EMA20, H={h:.2f}"
            )
        elif vol_ok and short_ok:
            risk = or_high - price
            setup.signal, setup.entry, setup.stop = "SHORT", price, or_high
            setup.target = price - config.RISK_REWARD * risk
            setup.reasons.append(
                f"ORB short: price {price:.2f} below OR low {or_low:.2f}, below "
                f"VWAP {cur_vwap:.2f}, EMA9<EMA20, H={h:.2f}"
            )
        else:
            setup.reasons.append(
                "no breakout confirmation: need price beyond opening range on the "
                "VWAP side with EMA alignment"
                + (" and 1.5x relative volume (mixed regime)" if need_extra else "")
            )

    elif playbook == "REVERT_VWAP_FADE":
        stretched_up = price > cur_vwap + config.VWAP_FADE_SIGMA * dev_sigma
        stretched_dn = price < cur_vwap - config.VWAP_FADE_SIGMA * dev_sigma
        if stretched_up and cur_rsi >= config.RSI_OVERBOUGHT:
            session_high = float(session["High"].max())
            setup.signal, setup.entry = "SHORT", price
            setup.stop, setup.target = session_high, cur_vwap
            setup.reasons.append(
                f"VWAP fade short: {((price - cur_vwap) / dev_sigma):.1f} sigma above "
                f"VWAP, RSI {cur_rsi:.0f} overbought, mean-reverting H={h:.2f}"
            )
        elif stretched_dn and cur_rsi <= config.RSI_OVERSOLD:
            session_low = float(session["Low"].min())
            setup.signal, setup.entry = "LONG", price
            setup.stop, setup.target = session_low, cur_vwap
            setup.reasons.append(
                f"VWAP fade long: {((cur_vwap - price) / dev_sigma):.1f} sigma below "
                f"VWAP, RSI {cur_rsi:.0f} oversold, mean-reverting H={h:.2f}"
            )
        else:
            setup.reasons.append(
                "no fade: price within 2 sigma of VWAP or RSI not at an extreme"
            )

    return setup


def _rel_volume_from_daily(daily: pd.DataFrame) -> float:
    vol20 = daily["Volume"].iloc[-21:-1].mean()
    return float(daily["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0


# Register at import time.
from . import register  # noqa: E402
from .base import DataRequirements, TradingMethod  # noqa: E402

METHOD = register(TradingMethod(
    key=KEY,
    name="Hurst-Regime Opening-Range / VWAP-Fade",
    paper="Optimizing Returns Using the Hurst Exponent and Q-Learning on "
          "Momentum and Mean Reversion Strategies",
    paper_url="https://arxiv.org/pdf/2205.11122",
    regime_applicability="TRENDING (ORB momentum) and MEAN_REVERTING (VWAP fade)",
    timeframe="intraday",
    data=DataRequirements(daily_period="9mo", needs_intraday=True),
    description=(
        "Daily Hurst exponent classifies the regime, then applies a 15-minute "
        "opening-range breakout in trending names (VWAP + EMA9/EMA20 confirmed) "
        "or a >2-sigma VWAP fade with an RSI extreme in mean-reverting names. "
        "Mixed/random-walk regimes require elevated relative volume to act."
    ),
    analyze=analyze,
))
