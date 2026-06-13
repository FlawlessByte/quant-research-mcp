"""Backtest engines: exact stats math + replay on synthetic series."""

import numpy as np

from quant_research_mcp.backtest import (
    Trade,
    _apply_costs,
    backtest_daily_method,
    compute_stats,
)
from tests.conftest import make_daily


def _trade(r: float, risk: float = 1.0, direction: str = "LONG",
           day: str = "2024-06-01") -> Trade:
    entry = 100.0
    exit_ = entry + r * risk if direction == "LONG" else entry - r * risk
    return Trade(ticker="T", direction=direction, entry_date=day, exit_date=day,
                 entry=entry, exit=exit_, risk_per_share=risk, exit_reason="t")


def test_r_multiple_signs():
    assert _trade(2.0).r_multiple == 2.0
    assert _trade(-1.0).r_multiple == -1.0
    assert _trade(2.0, direction="SHORT").r_multiple == 2.0


def test_compute_stats_exact():
    trades = [_trade(2.0), _trade(2.0), _trade(-1.0), _trade(-1.0)]
    s = compute_stats(trades, risk_pct=0.01, start_equity=100_000)
    assert s["n_trades"] == 4
    assert s["win_rate"] == 0.5
    assert s["avg_r"] == 0.5
    assert s["profit_factor"] == 2.0
    # equity: 1.02 * 1.02 * 0.99 * 0.99 compounded on 100k
    expected = 100_000 * 1.02 * 1.02 * 0.99 * 0.99
    assert abs(s["ending_equity"] - round(expected, 2)) < 0.01


def test_compute_stats_empty():
    assert compute_stats([])["n_trades"] == 0


def test_apply_costs_direction():
    assert _apply_costs(100.0, "buy", 10) == 100.1     # pay up
    assert _apply_costs(100.0, "sell", 10) == 99.9     # give up


def test_donchian_replay_catches_trend():
    """Flat 300 bars then a strong 200-bar ramp -> at least one LONG trade
    that exits profitably on the trail."""
    rng = np.random.default_rng(11)
    flat = 100 + rng.normal(0, 0.4, 300)
    ramp = np.linspace(100, 220, 200) + rng.normal(0, 0.8, 200)
    df = make_daily(np.concatenate([flat, ramp]))
    trades = backtest_daily_method("donchian_trend", "T", df, costs_bps=5)
    assert len(trades) >= 1
    longs = [t for t in trades if t.direction == "LONG"]
    assert longs, "expected at least one long in a ramp"
    assert max(t.r_multiple for t in longs) > 1.0


def test_rsi2_replay_generates_trades():
    """Choppy uptrend produces RSI(2) dips above the 200d SMA."""
    rng = np.random.default_rng(5)
    closes = [100.0]
    for i in range(600):
        drift = 0.15
        shock = -2.5 if i % 40 in (0, 1, 2) else rng.normal(0, 0.6)
        closes.append(max(closes[-1] + drift + shock, 5.0))
    df = make_daily(np.array(closes))
    trades = backtest_daily_method("rsi2_reversion", "T", df, costs_bps=5)
    assert len(trades) >= 3
    stats = compute_stats(trades)
    assert stats["n_trades"] == len(trades)
