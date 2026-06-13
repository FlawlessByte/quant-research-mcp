"""Method registry contract.

A `TradingMethod` wraps one paper-backed strategy: metadata for discovery plus
an `analyze` callable that turns market data into a `TradeSetup`. Future papers
become new modules that build a `TradingMethod` and call `register(...)`.

Timeframes: "intraday" (day trades, needs 5m session data), "swing" (days),
"position" (weeks/months). Daily-only methods set needs_intraday=False and the
server skips the intraday download. Cross-sectional methods (ranking a
universe, pairs) set universe_based=True and implement `analyze_universe`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataRequirements:
    daily_period: str = "9mo"        # history needed for the daily frame
    needs_intraday: bool = False     # fetch 5m session data?
    universe_based: bool = False     # analyze a list of tickers together?
    min_tickers: int = 1             # for universe methods (pairs -> exactly 2)
    max_tickers: int = 1


@dataclass
class TradeSetup:
    """Output of a method's analysis for a single ticker (or pair leg)."""
    ticker: str
    method_key: str
    signal: str                  # LONG / SHORT / NO_ENTRY
    playbook: str                # method-specific sub-strategy label
    regime: str                  # TRENDING / MEAN_REVERTING / RANDOM_WALK / N/A
    hurst: float | None
    price: float
    atr_daily: float
    rel_volume: float
    rsi: float
    timeframe: str = "intraday"  # intraday / swing / position
    holding_period_hint: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    extras: dict = field(default_factory=dict)   # method-specific fields
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "method_key": self.method_key,
            "signal": self.signal,
            "playbook": self.playbook,
            "regime": self.regime,
            "timeframe": self.timeframe,
            "holding_period_hint": self.holding_period_hint or None,
            "hurst": round(self.hurst, 3) if self.hurst is not None else None,
            "price": round(self.price, 2),
            "atr_daily": round(self.atr_daily, 2),
            "rel_volume": round(self.rel_volume, 2),
            "rsi": round(self.rsi, 1),
            "reasons": self.reasons,
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in self.extras.items()},
        }
        for k in ("entry", "stop", "target"):
            v = getattr(self, k)
            d[k] = round(v, 2) if v is not None else None
        return {k: v for k, v in d.items() if v is not None or k in
                ("entry", "stop", "target", "hurst")}


# analyze(daily_df, intraday_session_df_or_None, context) -> TradeSetup
AnalyzeFn = Callable[[pd.DataFrame, pd.DataFrame | None, dict], TradeSetup]
# analyze_universe({ticker: daily_df}, context) -> list[TradeSetup]
AnalyzeUniverseFn = Callable[[dict[str, pd.DataFrame], dict], list[TradeSetup]]


@dataclass
class TradingMethod:
    key: str                     # stable id, e.g. "hurst_regime_orb"
    name: str                    # human title
    paper: str                   # citation
    paper_url: str               # arXiv/DOI link
    regime_applicability: str    # which regimes this method targets
    description: str             # one-paragraph summary of the mechanics
    timeframe: str = "intraday"  # intraday / swing / position
    data: DataRequirements = field(default_factory=DataRequirements)
    analyze: AnalyzeFn | None = None
    analyze_universe: AnalyzeUniverseFn | None = None

    def metadata(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "paper": self.paper,
            "paper_url": self.paper_url,
            "regime_applicability": self.regime_applicability,
            "timeframe": self.timeframe,
            "universe_based": self.data.universe_based,
            "description": self.description,
        }
