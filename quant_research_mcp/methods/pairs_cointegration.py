"""Pairs trading via cointegration (Gatev et al. 2006; Engle & Granger 1987).

Universe method requiring exactly two tickers (A, B):

1. OLS hedge ratio beta: log(A) = alpha + beta*log(B) over the lookback.
2. ADF test on the residual spread; tradeable only if stationary (p < 0.05).
3. Z-score of the current spread vs the lookback distribution.
   - z > +2: spread rich -> SHORT A / LONG beta*B
   - z < -2: spread cheap -> LONG A / SHORT beta*B
   - exit at z = 0, stop at |z| = 3.5.

Two TradeSetups are returned (one per leg) sharing the pair statistics.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .. import config
from ..indicators import atr, rsi
from .base import DataRequirements, TradeSetup, TradingMethod

KEY = "pairs_cointegration"


def _pair_stats(a: pd.Series, b: pd.Series) -> dict:
    la, lb = np.log(a.values), np.log(b.values)
    x = np.column_stack([np.ones_like(lb), lb])
    beta_vec, *_ = np.linalg.lstsq(x, la, rcond=None)
    alpha, beta = float(beta_vec[0]), float(beta_vec[1])
    spread = la - (alpha + beta * lb)
    adf_p = float(adfuller(spread, autolag="AIC")[1])
    mu, sigma = float(spread.mean()), float(spread.std())
    z = float((spread[-1] - mu) / sigma) if sigma > 0 else 0.0
    return {"alpha": alpha, "beta": beta, "adf_pvalue": adf_p,
            "spread_mean": mu, "spread_sigma": sigma, "zscore": z}


def _leg_setup(ticker: str, df: pd.DataFrame, signal: str, stats: dict,
               role: str, partner: str) -> TradeSetup:
    closes = df["Close"].dropna()
    price = float(closes.iloc[-1])
    atr_daily = float(atr(df).iloc[-1])
    vol20 = df["Volume"].iloc[-21:-1].mean()
    setup = TradeSetup(
        ticker=ticker, method_key=KEY, signal=signal,
        playbook="PAIRS_ZSCORE", regime="MEAN_REVERTING",
        hurst=None, price=price, atr_daily=atr_daily,
        rel_volume=float(df["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0,
        rsi=float(rsi(closes).iloc[-1]), timeframe="swing",
        holding_period_hint="days to weeks, exit when spread z-score reverts to 0",
        extras={**stats, "leg_role": role, "pair_with": partner},
    )
    if signal != "NO_ENTRY":
        z, sig = stats["zscore"], stats["spread_sigma"]
        # Map z-score levels to this leg's price space via its own volatility:
        # stop when the spread moves (STOP_Z - |z|) more sigmas against us.
        adverse_sigmas = max(config.PAIRS_STOP_Z - abs(z), 0.5)
        move = adverse_sigmas * sig * price            # log-spread sigma ~ pct move
        setup.entry = price
        setup.stop = price + move if signal == "SHORT" else price - move
        favorable = abs(z) * sig * price               # back to spread mean
        setup.target = price - favorable if signal == "SHORT" else price + favorable
    return setup


def analyze_universe(daily_map: dict[str, pd.DataFrame], context: dict
                     ) -> list[TradeSetup]:
    if len(daily_map) != 2:
        raise ValueError("pairs_cointegration requires exactly 2 tickers")
    (ta, da), (tb, db) = daily_map.items()
    joined = pd.concat([da["Close"], db["Close"]], axis=1, keys=[ta, tb]).dropna()
    joined = joined.iloc[-config.PAIRS_LOOKBACK:]
    if len(joined) < 60:
        raise ValueError(f"need >=60 overlapping daily bars, have {len(joined)}")

    stats = _pair_stats(joined[ta], joined[tb])
    z, p = stats["zscore"], stats["adf_pvalue"]

    if p > config.PAIRS_ADF_PVALUE:
        sig_a = sig_b = "NO_ENTRY"
        reason = (f"not cointegrated: ADF p={p:.3f} > {config.PAIRS_ADF_PVALUE} — "
                  f"spread non-stationary, no statistical basis to trade")
    elif z >= config.PAIRS_ENTRY_Z:
        sig_a, sig_b = "SHORT", "LONG"
        reason = (f"spread rich: z={z:+.2f} >= +{config.PAIRS_ENTRY_Z} — short "
                  f"{ta}, long {stats['beta']:.2f}x {tb}; exit z=0, stop |z|="
                  f"{config.PAIRS_STOP_Z}")
    elif z <= -config.PAIRS_ENTRY_Z:
        sig_a, sig_b = "LONG", "SHORT"
        reason = (f"spread cheap: z={z:+.2f} <= -{config.PAIRS_ENTRY_Z} — long "
                  f"{ta}, short {stats['beta']:.2f}x {tb}; exit z=0, stop |z|="
                  f"{config.PAIRS_STOP_Z}")
    else:
        sig_a = sig_b = "NO_ENTRY"
        reason = (f"cointegrated (ADF p={p:.3f}) but |z|={abs(z):.2f} < entry "
                  f"{config.PAIRS_ENTRY_Z} — wait for a wider spread")

    leg_a = _leg_setup(ta, da, sig_a, stats, "primary", tb)
    leg_b = _leg_setup(tb, db, sig_b, stats, "hedge", ta)
    leg_a.reasons.append(reason)
    leg_b.reasons.append(reason)
    return [leg_a, leg_b]


from . import register  # noqa: E402

METHOD = register(TradingMethod(
    key=KEY,
    name="Pairs Trading (Engle-Granger Cointegration)",
    paper="Gatev, Goetzmann & Rouwenhorst, 'Pairs Trading: Performance of a "
          "Relative-Value Arbitrage Rule', RFS 19 (2006); Engle & Granger (1987)",
    paper_url="https://doi.org/10.1093/rfs/hhj020",
    regime_applicability="MEAN_REVERTING spread between two cointegrated names",
    timeframe="swing",
    data=DataRequirements(daily_period="2y", needs_intraday=False,
                          universe_based=True, min_tickers=2, max_tickers=2),
    description=(
        "Estimates an OLS hedge ratio between two names, verifies the spread is "
        "stationary with an ADF test, and trades z-score extremes of the spread: "
        "enter at |z|>=2 (short the rich leg, long the cheap leg weighted by "
        "beta), exit at z=0, stop at |z|=3.5. Market-neutral by construction."
    ),
    analyze_universe=analyze_universe,
))
