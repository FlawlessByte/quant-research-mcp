"""Backtesting engines.

Architecture: the engines REPLAY each method's own `analyze()` over history,
so the backtest executes exactly the live signal logic — there is no separate
signal implementation to drift out of sync.

Engines:
- Daily engine: per-ticker swing/position methods (donchian_trend,
  rsi2_reversion, and any future daily method). Entry at next bar open after a
  signal; exits via method-specific handlers (trail/MA-cross) with a generic
  stop/target fallback.
- Intraday engine: hurst_regime_orb. Replays each available 5m session,
  entering when the growing-session analyze fires, managing stop/target within
  the session, flat by the close.
- Portfolio engine: xs_momentum. Monthly re-rank, hold top N equal weight.

Costs: `costs_bps` one-way applied to entry and exit prices.
All results are deterministic given the same input data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .indicators import atr, sma
from .methods import get as get_method


@dataclass
class Trade:
    ticker: str
    direction: str               # LONG / SHORT
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    risk_per_share: float
    exit_reason: str

    @property
    def r_multiple(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        sign = 1.0 if self.direction == "LONG" else -1.0
        return sign * (self.exit - self.entry) / self.risk_per_share

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker, "direction": self.direction,
            "entry_date": self.entry_date, "exit_date": self.exit_date,
            "entry": round(self.entry, 2), "exit": round(self.exit, 2),
            "r_multiple": round(self.r_multiple, 3),
            "exit_reason": self.exit_reason,
        }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(dd.min()) if len(dd) else 0.0


def compute_stats(trades: list[Trade], risk_pct: float = 0.01,
                  start_equity: float = config.BACKTEST_EQUITY) -> dict:
    """Pooled trade stats + fixed-fractional compounded equity curve."""
    if not trades:
        return {"n_trades": 0, "note": "no trades generated"}
    trades = sorted(trades, key=lambda t: t.exit_date)
    rs = np.array([t.r_multiple for t in trades])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    equity = [start_equity]
    for r in rs:
        equity.append(equity[-1] * (1 + risk_pct * r))
    eq = np.array(equity)
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    return {
        "n_trades": len(trades),
        "win_rate": round(float(len(wins) / len(rs)), 3),
        "avg_r": round(float(rs.mean()), 3),
        "expectancy_r": round(float(rs.mean()), 3),
        "best_r": round(float(rs.max()), 2),
        "worst_r": round(float(rs.min()), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown_pct": round(_drawdown(eq) * 100, 2),
        "ending_equity": round(float(eq[-1]), 2),
        "total_return_pct": round((float(eq[-1]) / start_equity - 1) * 100, 2),
        "risk_pct_per_trade": risk_pct * 100,
    }


def _split_stats(trades: list[Trade], risk_pct: float) -> dict:
    """In-sample / out-of-sample halves by exit date, for stability read."""
    if len(trades) < 6:
        return {}
    trades = sorted(trades, key=lambda t: t.exit_date)
    mid = len(trades) // 2
    return {
        "in_sample_first_half": compute_stats(trades[:mid], risk_pct),
        "out_of_sample_second_half": compute_stats(trades[mid:], risk_pct),
    }


def _apply_costs(price: float, side: str, costs_bps: float) -> float:
    """Worse fill by costs_bps: pay up on buys, give up on sells."""
    adj = price * costs_bps / 10_000
    return price + adj if side == "buy" else price - adj


# ---------------------------------------------------------------------------
# Daily engine (per-ticker swing/position methods)
# ---------------------------------------------------------------------------
def _exit_donchian(position: dict, window: pd.DataFrame) -> tuple[float, str] | None:
    """Trail: 20d opposite channel or 2.5x ATR, whichever is tighter-favourable."""
    n_x = config.DONCHIAN_EXIT
    cur_atr = float(atr(window).iloc[-1])
    close = float(window["Close"].iloc[-1])
    bar = window.iloc[-1]
    if position["direction"] == "LONG":
        trail = max(float(window["Low"].iloc[-(n_x + 1):-1].min()),
                    close - config.DONCHIAN_ATR_TRAIL * cur_atr)
        position["stop"] = max(position["stop"], trail)
        if float(bar["Low"]) <= position["stop"]:
            return min(float(bar["Open"]), position["stop"]), "trail_stop"
    else:
        trail = min(float(window["High"].iloc[-(n_x + 1):-1].max()),
                    close + config.DONCHIAN_ATR_TRAIL * cur_atr)
        position["stop"] = min(position["stop"], trail)
        if float(bar["High"]) >= position["stop"]:
            return max(float(bar["Open"]), position["stop"]), "trail_stop"
    return None


def _exit_rsi2(position: dict, window: pd.DataFrame) -> tuple[float, str] | None:
    """Stop hit, else close crossing the 5d SMA exits at that close."""
    bar = window.iloc[-1]
    close = float(bar["Close"])
    ma5 = float(sma(window["Close"], config.RSI2_EXIT_MA).iloc[-1])
    if position["direction"] == "LONG":
        if float(bar["Low"]) <= position["stop"]:
            return min(float(bar["Open"]), position["stop"]), "stop"
        if close > ma5:
            return close, "ma5_cross"
    else:
        if float(bar["High"]) >= position["stop"]:
            return max(float(bar["Open"]), position["stop"]), "stop"
        if close < ma5:
            return close, "ma5_cross"
    return None


def _exit_generic(position: dict, window: pd.DataFrame) -> tuple[float, str] | None:
    """Stop / target intrabar; stop checked first (conservative)."""
    bar = window.iloc[-1]
    lo, hi, op = float(bar["Low"]), float(bar["High"]), float(bar["Open"])
    if position["direction"] == "LONG":
        if lo <= position["stop"]:
            return min(op, position["stop"]), "stop"
        if position.get("target") and hi >= position["target"]:
            return max(op, position["target"]), "target"
    else:
        if hi >= position["stop"]:
            return max(op, position["stop"]), "stop"
        if position.get("target") and lo <= position["target"]:
            return min(op, position["target"]), "target"
    return None


_EXIT_HANDLERS = {
    "donchian_trend": _exit_donchian,
    "rsi2_reversion": _exit_rsi2,
}


def backtest_daily_method(method_key: str, ticker: str, daily: pd.DataFrame,
                          costs_bps: float = config.BACKTEST_COSTS_BPS,
                          warmup: int = 260) -> list[Trade]:
    """Replay a per-ticker daily method bar by bar."""
    method = get_method(method_key)
    if method.analyze is None:
        raise ValueError(f"{method_key} is universe-based; not supported here")
    exit_fn = _EXIT_HANDLERS.get(method_key, _exit_generic)
    trades: list[Trade] = []
    position: dict | None = None

    bars = daily.dropna(subset=["Close"])
    for i in range(warmup, len(bars) - 1):
        window = bars.iloc[: i + 1]
        next_bar = bars.iloc[i + 1]
        next_date = str(bars.index[i + 1].date())

        if position is not None:
            # manage on the NEXT bar (we act after observing the signal bar)
            managed = exit_fn(position, bars.iloc[: i + 2])
            if managed is not None:
                raw_exit, reason = managed
                side = "sell" if position["direction"] == "LONG" else "buy"
                trades.append(Trade(
                    ticker=ticker, direction=position["direction"],
                    entry_date=position["entry_date"], exit_date=next_date,
                    entry=position["entry"],
                    exit=_apply_costs(raw_exit, side, costs_bps),
                    risk_per_share=position["risk"], exit_reason=reason,
                ))
                position = None
            continue

        setup = method.analyze(window, None, {"ticker": ticker})
        if setup.signal in ("LONG", "SHORT") and setup.stop is not None:
            side = "buy" if setup.signal == "LONG" else "sell"
            entry = _apply_costs(float(next_bar["Open"]), side, costs_bps)
            risk = abs(entry - setup.stop)
            if risk <= 0:
                continue
            position = {
                "direction": setup.signal, "entry": entry, "stop": setup.stop,
                "target": setup.target, "risk": risk, "entry_date": next_date,
            }

    if position is not None:  # close any open position at the last close
        last = bars.iloc[-1]
        side = "sell" if position["direction"] == "LONG" else "buy"
        trades.append(Trade(
            ticker=ticker, direction=position["direction"],
            entry_date=position["entry_date"], exit_date=str(bars.index[-1].date()),
            entry=position["entry"],
            exit=_apply_costs(float(last["Close"]), side, costs_bps),
            risk_per_share=position["risk"], exit_reason="end_of_data",
        ))
    return trades


# ---------------------------------------------------------------------------
# Intraday engine (hurst_regime_orb)
# ---------------------------------------------------------------------------
def backtest_intraday_method(method_key: str, ticker: str, daily: pd.DataFrame,
                             intraday: pd.DataFrame,
                             costs_bps: float = config.BACKTEST_COSTS_BPS
                             ) -> list[Trade]:
    """Replay each 5m session: enter when the growing-session analyze fires."""
    method = get_method(method_key)
    if method.analyze is None:
        raise ValueError(f"{method_key} is universe-based; not supported here")
    trades: list[Trade] = []
    if intraday.empty:
        return trades

    sessions = intraday.groupby([ts.date() for ts in intraday.index])
    daily_dates = [d.date() for d in daily.index]

    for day, bars in sessions:
        # daily history strictly BEFORE this session (no lookahead)
        cut = next((j for j, d in enumerate(daily_dates) if d >= day), len(daily_dates))
        hist = daily.iloc[:cut]
        if len(hist) < config.HURST_WINDOW:
            continue
        vol20 = hist["Volume"].iloc[-20:].mean()
        day_vol = float(bars["Volume"].sum())
        rel_volume = day_vol / vol20 if vol20 > 0 else 1.0
        ctx = {"ticker": ticker, "rel_volume": rel_volume}

        position: dict | None = None
        traded_this_session = False   # one setup per session, per the playbook
        n_or = config.OPENING_RANGE_BARS
        for k in range(n_or + 1, len(bars)):
            window = bars.iloc[:k]
            bar = bars.iloc[k]
            ts = str(bars.index[k])

            if position is None:
                if traded_this_session:
                    break
                setup = method.analyze(hist, window, ctx)
                if setup.signal in ("LONG", "SHORT") and setup.stop is not None:
                    side = "buy" if setup.signal == "LONG" else "sell"
                    entry = _apply_costs(float(bar["Open"]), side, costs_bps)
                    risk = abs(entry - setup.stop)
                    # skip degenerate setups where the stop is on top of entry
                    if risk >= entry * 0.001:
                        position = {"direction": setup.signal, "entry": entry,
                                    "stop": setup.stop, "target": setup.target,
                                    "risk": risk, "entry_date": ts}
                        traded_this_session = True
                continue

            done = _exit_generic(position, bars.iloc[: k + 1])
            if done is not None:
                raw_exit, reason = done
                side = "sell" if position["direction"] == "LONG" else "buy"
                trades.append(Trade(
                    ticker=ticker, direction=position["direction"],
                    entry_date=position["entry_date"], exit_date=ts,
                    entry=position["entry"],
                    exit=_apply_costs(raw_exit, side, costs_bps),
                    risk_per_share=position["risk"], exit_reason=reason,
                ))
                position = None

        if position is not None:  # flat by the close
            last = bars.iloc[-1]
            side = "sell" if position["direction"] == "LONG" else "buy"
            trades.append(Trade(
                ticker=ticker, direction=position["direction"],
                entry_date=position["entry_date"], exit_date=str(bars.index[-1]),
                entry=position["entry"],
                exit=_apply_costs(float(last["Close"]), side, costs_bps),
                risk_per_share=position["risk"], exit_reason="session_close",
            ))
    return trades


# ---------------------------------------------------------------------------
# Portfolio engine (xs_momentum)
# ---------------------------------------------------------------------------
def backtest_xs_momentum(daily_map: dict[str, pd.DataFrame], top_n: int,
                         costs_bps: float = config.BACKTEST_COSTS_BPS) -> dict:
    """Monthly re-rank; hold top N equal weight the following month."""
    closes = pd.concat({t: df["Close"] for t, df in daily_map.items()}, axis=1)
    closes = closes.dropna(how="all").ffill()
    month_ends = closes.groupby([d.strftime("%Y-%m") for d in closes.index]).tail(1).index

    look, skip = config.XS_MOM_LOOKBACK, config.XS_MOM_SKIP
    monthly_returns: list[float] = []
    dates: list[str] = []
    turnover_cost = 2 * costs_bps / 10_000   # round-trip on rebalance

    prev_book: set[str] = set()
    for m in range(len(month_ends) - 1):
        asof, nxt = month_ends[m], month_ends[m + 1]
        upto = closes.loc[:asof]
        if len(upto) < look + 1:
            continue
        mom = (upto.iloc[-skip - 1] / upto.iloc[-look - 1] - 1).dropna()
        if len(mom) < top_n:
            continue
        book = set(mom.sort_values(ascending=False).head(top_n).index)
        seg = closes.loc[asof:nxt, sorted(book)]
        rets = (seg.iloc[-1] / seg.iloc[0] - 1).fillna(0.0)
        gross = float(rets.mean())
        changed = len(book - prev_book) / max(top_n, 1)
        net = gross - changed * turnover_cost
        monthly_returns.append(net)
        dates.append(str(nxt.date()))
        prev_book = book

    if not monthly_returns:
        return {"n_periods": 0, "note": "not enough history for a single rebalance"}
    rets = np.array(monthly_returns)
    eq = np.cumprod(1 + rets)
    years = len(rets) / 12
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0.0
    half = len(rets) // 2
    return {
        "n_periods": len(rets),
        "monthly_mean_pct": round(float(rets.mean()) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(_drawdown(eq) * 100, 2),
        "total_return_pct": round((float(eq[-1]) - 1) * 100, 2),
        "win_months_pct": round(float((rets > 0).mean()) * 100, 1),
        "is_oos": {
            "first_half_mean_monthly_pct": round(float(rets[:half].mean()) * 100, 2),
            "second_half_mean_monthly_pct": round(float(rets[half:].mean()) * 100, 2),
        } if half >= 3 else {},
        "equity_curve": [round(float(v), 4) for v in eq[:: max(1, len(eq) // 100)]],
        "dates_sampled": dates[:: max(1, len(dates) // 100)],
    }


# ---------------------------------------------------------------------------
# Entry point used by the server tool
# ---------------------------------------------------------------------------
def run_backtest(method_key: str, tickers: list[str],
                 daily_map: dict[str, pd.DataFrame],
                 intraday_map: dict[str, pd.DataFrame],
                 costs_bps: float = config.BACKTEST_COSTS_BPS,
                 risk_pct: float = 0.01, top_n: int = config.XS_MOM_TOP_N) -> dict:
    method = get_method(method_key)

    if method_key == "xs_momentum":
        return {"method": method_key, "engine": "portfolio_monthly_rebalance",
                "tickers": tickers, "costs_bps": costs_bps,
                **backtest_xs_momentum(daily_map, top_n, costs_bps)}

    if method_key == "pairs_cointegration":
        return {"method": method_key,
                "error": "backtest not yet supported for pairs (rolling "
                         "re-estimation engine planned); use quant_analyze_universe "
                         "for the current spread state"}

    trades: list[Trade] = []
    for t in tickers:
        daily = daily_map.get(t)
        if daily is None or daily.empty:
            continue
        if method.timeframe == "intraday":
            intraday = intraday_map.get(t)
            if intraday is None or intraday.empty:
                continue
            trades += backtest_intraday_method(method_key, t, daily, intraday,
                                               costs_bps)
        else:
            trades += backtest_daily_method(method_key, t, daily, costs_bps)

    out = {
        "method": method_key,
        "engine": "intraday_session_replay" if method.timeframe == "intraday"
                  else "daily_bar_replay",
        "tickers": tickers, "costs_bps": costs_bps,
        "stats": compute_stats(trades, risk_pct),
        "is_oos": _split_stats(trades, risk_pct),
        "last_trades": [t.as_dict() for t in
                        sorted(trades, key=lambda x: x.exit_date)[-10:]],
    }
    return out
