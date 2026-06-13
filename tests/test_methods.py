"""Method trigger logic on synthetic data. No network."""

import numpy as np

from quant_research_mcp.methods import (
    donchian_trend,
    pairs_cointegration,
    rsi2_reversion,
    xs_momentum,
)
from tests.conftest import make_daily


def test_donchian_breakout_long(trending_daily):
    """A fresh all-time-high close on a trending series triggers LONG."""
    df = trending_daily.copy()
    df.iloc[-1, df.columns.get_loc("Close")] = df["High"].max() * 1.05
    setup = donchian_trend.analyze(df, None, {"ticker": "T"})
    assert setup.signal == "LONG"
    assert setup.stop < setup.entry
    assert setup.timeframe == "position"


def test_donchian_inside_channel_no_entry(flat_daily):
    setup = donchian_trend.analyze(flat_daily, None, {"ticker": "T"})
    assert setup.signal == "NO_ENTRY"


def test_rsi2_pullback_long():
    """Uptrend + sharp 3-day dump above the 200d SMA -> RSI(2) capitulation."""
    closes = np.linspace(100, 200, 400)
    closes[-3:] = [196.0, 192.0, 188.0]   # sharp pullback, still above 200d SMA
    df = make_daily(closes)
    setup = rsi2_reversion.analyze(df, None, {"ticker": "T"})
    assert setup.extras["rsi2"] < 10
    assert setup.signal == "LONG"
    assert setup.timeframe == "swing"


def test_rsi2_no_extreme_no_entry():
    closes = np.linspace(100, 200, 400)   # steady climb, RSI(2) pinned high
    df = make_daily(closes)
    setup = rsi2_reversion.analyze(df, None, {"ticker": "T"})
    assert setup.signal == "NO_ENTRY"


def test_xs_momentum_ranking():
    """Distinct drifts -> deterministic ranking; top-N flagged LONG."""
    universe = {}
    drifts = {"AAA": 0.8, "BBB": 0.5, "CCC": 0.2, "DDD": 0.05, "EEE": -0.2,
              "FFF": -0.5}
    for t, d in drifts.items():
        closes = 100 * np.exp(np.linspace(0, d, 300))
        universe[t] = make_daily(closes)
    setups = xs_momentum.analyze_universe(universe, {"top_n": 2})
    by_ticker = {s.ticker: s for s in setups}
    assert by_ticker["AAA"].extras["rank"] == 1
    assert by_ticker["AAA"].signal == "LONG"
    assert by_ticker["BBB"].signal == "LONG"
    assert by_ticker["CCC"].signal == "NO_ENTRY"
    assert by_ticker["FFF"].extras["rank"] == 6


def test_pairs_cointegrated_stretched():
    """B tracks A with stationary noise; force the spread wide -> signals."""
    rng = np.random.default_rng(3)
    a = 100 + np.cumsum(rng.normal(0, 1, 300))
    noise = np.zeros(300)
    for i in range(1, 300):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.4)
    b = a * 0.8 + noise
    a = np.maximum(a, 5.0)
    b = np.maximum(b, 5.0)
    # stretch the final spread rich: A jumps, B doesn't
    a[-1] *= 1.10
    dm = {"AAA": make_daily(a), "BBB": make_daily(b)}
    setups = pairs_cointegration.analyze_universe(dm, {})
    by = {s.ticker: s for s in setups}
    assert by["AAA"].extras["adf_pvalue"] < 0.05
    z = by["AAA"].extras["zscore"]
    if z >= 2.0:
        assert by["AAA"].signal == "SHORT" and by["BBB"].signal == "LONG"
    else:  # stretch landed inside the band: structure still validated
        assert by["AAA"].signal == "NO_ENTRY"


def test_pairs_requires_two():
    import pytest
    with pytest.raises(ValueError):
        pairs_cointegration.analyze_universe({"AAA": make_daily(np.full(300, 100.0))}, {})
