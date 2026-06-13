"""Cross-sectional 12-1 momentum (Jegadeesh & Titman 1993).

Universe method on daily bars: rank every ticker by its trailing 12-month
return excluding the most recent month (252d window, skip last 21d — the
short-term reversal), go long the top N. Re-rank monthly; this tool reports
today's ranking and flags names entering/holding the long book.

Each returned TradeSetup is one ranked name; entry = current price, stop =
2.5x ATR (volatility-scaled, per common practitioner implementations),
target = 2R reference.
"""

import pandas as pd

from .. import config
from ..indicators import atr, hurst_exponent, rsi
from .base import DataRequirements, TradeSetup, TradingMethod

KEY = "xs_momentum"


def _momentum_12_1(closes: pd.Series) -> float | None:
    need = config.XS_MOM_LOOKBACK + 1
    if len(closes) < need:
        return None
    past = closes.iloc[-config.XS_MOM_LOOKBACK - 1]
    recent = closes.iloc[-config.XS_MOM_SKIP - 1]
    if past <= 0:
        return None
    return float(recent / past - 1)


def analyze_universe(daily_map: dict[str, pd.DataFrame], context: dict
                     ) -> list[TradeSetup]:
    scores: list[tuple[str, float]] = []
    for t, df in daily_map.items():
        closes = df["Close"].dropna()
        m = _momentum_12_1(closes)
        if m is not None:
            scores.append((t, m))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_n = int(context.get("top_n", config.XS_MOM_TOP_N))

    setups: list[TradeSetup] = []
    for rank, (t, m) in enumerate(scores, 1):
        in_book = rank <= top_n
        df = daily_map[t]
        closes = df["Close"].dropna()
        price = float(closes.iloc[-1])
        atr_daily = float(atr(df).iloc[-1])
        h = hurst_exponent(closes.iloc[-config.HURST_WINDOW:], config.HURST_MAX_LAG)
        vol20 = df["Volume"].iloc[-21:-1].mean()
        rel_volume = float(df["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

        setup = TradeSetup(
            ticker=t, method_key=KEY,
            signal="LONG" if in_book else "NO_ENTRY",
            playbook="XS_MOMENTUM_12_1",
            regime="TRENDING" if h >= config.HURST_TREND else
                   ("MEAN_REVERTING" if h <= config.HURST_REVERT else "RANDOM_WALK"),
            hurst=h, price=price, atr_daily=atr_daily, rel_volume=rel_volume,
            rsi=float(rsi(closes).iloc[-1]), timeframe="position",
            holding_period_hint="hold until monthly re-rank drops it from the top book",
            extras={"momentum_12_1": round(m, 4), "rank": rank,
                    "universe_size": len(scores), "top_n": top_n},
        )
        if in_book:
            stop = price - 2.5 * atr_daily
            setup.entry, setup.stop = price, stop
            setup.target = price + config.RISK_REWARD * (price - stop)
            setup.reasons.append(
                f"rank {rank}/{len(scores)} by 12-1 momentum ({m:+.1%}) — in the "
                f"top-{top_n} long book; re-rank monthly"
            )
        else:
            setup.reasons.append(
                f"rank {rank}/{len(scores)} by 12-1 momentum ({m:+.1%}) — outside "
                f"the top-{top_n} book"
            )
        setups.append(setup)
    return setups


from . import register  # noqa: E402

METHOD = register(TradingMethod(
    key=KEY,
    name="Cross-Sectional 12-1 Momentum",
    paper="Jegadeesh & Titman, 'Returns to Buying Winners and Selling Losers: "
          "Implications for Stock Market Efficiency', JF 48 (1993)",
    paper_url="https://doi.org/10.1111/j.1540-6261.1993.tb04702.x",
    regime_applicability="TRENDING — cross-sectional relative strength",
    timeframe="position",
    data=DataRequirements(daily_period="2y", needs_intraday=False,
                          universe_based=True, min_tickers=5, max_tickers=200),
    description=(
        "Ranks the universe by trailing 12-month return excluding the last month "
        "and holds the top N names long, re-ranked monthly. The canonical "
        "momentum anomaly; works at the portfolio level, not per-name."
    ),
    analyze_universe=analyze_universe,
))
