"""Technical indicators: EMA, RSI, ATR, VWAP, Hurst exponent.

Pure functions over pandas Series/DataFrames. No I/O, fully deterministic.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # loss == 0: pure gains -> 100; no movement at all -> neutral 50
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. df needs High/Low/Close columns."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP. Pass bars from a single session."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (typical * df["Volume"]).cumsum() / cum_vol


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def prorated_rel_volume(intraday: pd.DataFrame, vol20: float) -> float:
    """Time-of-day-adjusted relative volume.

    Full-day relative volume (today vs the 20-day average) reads low all
    morning because the day isn't done. This compares today's cumulative
    volume against the cumulative volume of the PRIOR sessions in the same
    intraday frame at the same bar count, scaled by vol20 so the baseline is
    the 20-day norm rather than just last week's.

    Falls back to plain today/vol20 if there is no prior-session data.
    """
    if intraday.empty or vol20 <= 0:
        return 1.0
    last_day = intraday.index[-1].date()
    today = intraday[[ts.date() == last_day for ts in intraday.index]]
    prior = intraday[[ts.date() != last_day for ts in intraday.index]]
    today_cum = float(today["Volume"].sum())
    if prior.empty:
        return today_cum / vol20
    n_bars = len(today)
    by_day = prior.groupby([ts.date() for ts in prior.index])["Volume"]
    # average cumulative volume across prior sessions at the same elapsed bars
    cums = [float(g.iloc[:n_bars].sum()) for _, g in by_day if len(g) > 0]
    expected_at_elapsed = sum(cums) / len(cums) if cums else 0.0
    prior_full = [float(g.sum()) for _, g in by_day]
    avg_full = sum(prior_full) / len(prior_full) if prior_full else 0.0
    if expected_at_elapsed <= 0 or avg_full <= 0:
        return today_cum / vol20
    # expected fraction of a full day at this elapsed point, applied to vol20
    elapsed_fraction = expected_at_elapsed / avg_full
    expected_today = vol20 * elapsed_fraction
    return today_cum / expected_today if expected_today > 0 else 1.0


def hurst_exponent(prices, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent from the scaling of lagged differences.

    H > 0.5 -> trending (persistent), H < 0.5 -> mean-reverting
    (anti-persistent), H ~ 0.5 -> random walk. After arXiv:2205.11122.
    """
    ts = np.log(np.asarray(prices, dtype=float))
    if len(ts) < max_lag * 3:
        return 0.5
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 5:
        return 0.5
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(slope)
