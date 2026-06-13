#!/usr/bin/env python3
"""quant_research_mcp — MCP server for quantitative trading research.

Exposes a registry of paper-backed trading methods plus a deterministic,
no-LLM decision helper. The agent calling this server supplies judgement
(e.g. interpreting news into a sentiment signal); the server supplies
reproducible math: screening, indicators, regime detection, method signals
and a scored ENTRY/NO_ENTRY decision with position sizing and an execution plan.

Typical workflow (day trade):
    quant_market_context -> quant_screen_universe -> quant_analyze_setup ->
    quant_check_events + quant_fetch_news (agent forms sentiment) ->
    quant_portfolio_risk -> quant_score_decision.
Validation: quant_backtest_method before trusting any method's parameters.

stdio transport. Read-only; places no orders. NOT financial advice.
"""


from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from quant_research_mcp import backtest as bt
from quant_research_mcp import config, data, decision, market, methods, portfolio
from quant_research_mcp.formatting import ResponseFormat, to_json
from quant_research_mcp.indicators import (
    atr,
    ema,
    hurst_exponent,
    prorated_rel_volume,
    rsi,
    vwap,
)
from quant_research_mcp.screener import screen

mcp = FastMCP("quant_research_mcp")

_RO = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
_RO_NET = {**_RO, "openWorldHint": True}      # touches yfinance
_RO_PURE = {**_RO, "openWorldHint": False}    # pure computation


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _rel_volume(daily) -> float:
    vol20 = daily["Volume"].iloc[-21:-1].mean()
    return float(daily["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0


def _err(msg: str) -> str:
    return f"Error: {msg}"


# --------------------------------------------------------------------------
# Input models
# --------------------------------------------------------------------------
class FormatInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="'markdown' for human-readable or 'json' for machine-readable",
    )


class DescribeMethodInput(FormatInput):
    method_key: str = Field(..., description="Registered method key, e.g. 'hurst_regime_orb'",
                            min_length=1, max_length=80)


class ScreenInput(FormatInput):
    tickers: list[str] | None = Field(
        default=None, description="Universe to screen; defaults to the built-in 40-name list",
        max_length=200)
    top_n: int = Field(default=config.TOP_N, description="Number of top candidates to return",
                       ge=1, le=50)


class TickerInput(FormatInput):
    ticker: str = Field(..., description="US equity symbol, e.g. 'NVDA'",
                        min_length=1, max_length=12)


class AnalyzeInput(TickerInput):
    method_key: str = Field(default="hurst_regime_orb",
                            description="Registered method to run", min_length=1, max_length=80)


class ScoreDecisionInput(FormatInput):
    setup: dict = Field(..., description=(
        "A setup dict as returned by quant_analyze_setup (must include signal, "
        "entry, stop, target, hurst, rel_volume, rsi, atr_daily, and optionally "
        "vwap/or_high/or_low for a richer execution plan)"))
    equity: float = Field(default=100_000.0, description="Account equity in USD", gt=0)
    risk_pct: float = Field(default=config.DEFAULT_ACCOUNT_RISK_PCT,
                            description="Fraction of equity risked per trade (0.005 = 0.5%)",
                            gt=0, le=0.1)
    news_sentiment: str | None = Field(
        default=None, description=(
            "OPTIONAL caller-derived sentiment: 'bullish', 'bearish' or 'neutral'. "
            "The agent interprets news; this tool only consumes the verdict. Aligned "
            "news boosts the score; contradicting news with confidence>=0.5 vetoes."))
    news_confidence: float = Field(default=0.0,
                                   description="Confidence in the sentiment, 0..1", ge=0, le=1)
    days_to_earnings: int | None = Field(
        default=None, ge=0,
        description=("Days until the next earnings report (from quant_check_events). "
                     f"Vetoes swing/position entries within {config.EARNINGS_VETO_DAYS} "
                     "days; intraday gets a warning."))
    portfolio_heat_pct: float | None = Field(
        default=None, ge=0,
        description=("Current total open risk as %% of equity (from "
                     "quant_portfolio_risk). Vetoes when at/above the "
                     f"{config.MAX_PORTFOLIO_HEAT * 100:.0f}%% limit."))


class AnalyzeUniverseInput(FormatInput):
    method_key: str = Field(..., description=(
        "A universe-based method key, e.g. 'xs_momentum' (5+ tickers) or "
        "'pairs_cointegration' (exactly 2 tickers)"), min_length=1, max_length=80)
    tickers: list[str] | None = Field(
        default=None, max_length=200,
        description="Tickers to analyze together; defaults to the built-in universe")
    top_n: int = Field(default=config.XS_MOM_TOP_N, ge=1, le=50,
                       description="Book size for ranking methods (xs_momentum)")


class BacktestInput(FormatInput):
    method_key: str = Field(..., description="Method to backtest", min_length=1,
                            max_length=80)
    tickers: list[str] = Field(..., min_length=1, max_length=50,
                               description="Tickers to backtest over")
    period: str = Field(default="2y", description=(
        "Daily history window, e.g. '1y', '2y', '5y'. Intraday methods are "
        "capped by the provider's 5m history (~60 days)."))
    costs_bps: float = Field(default=config.BACKTEST_COSTS_BPS, ge=0, le=100,
                             description="One-way slippage+fees in basis points")
    risk_pct: float = Field(default=0.01, gt=0, le=0.1,
                            description="Fraction of equity risked per trade for the "
                                        "equity curve")
    top_n: int = Field(default=config.XS_MOM_TOP_N, ge=1, le=50,
                       description="Book size (xs_momentum only)")


class Position(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    ticker: str = Field(..., min_length=1, max_length=12)
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    entry: float = Field(..., gt=0)
    stop: float = Field(..., gt=0)
    shares: int = Field(..., ge=0)


class Candidate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    ticker: str = Field(..., min_length=1, max_length=12)
    direction: str = Field(..., pattern="^(LONG|SHORT)$")
    entry: float = Field(..., gt=0)
    stop: float = Field(..., gt=0)


class PortfolioRiskInput(FormatInput):
    positions: list[Position] = Field(..., max_length=50, description=(
        "Your current open positions. The server stores nothing — pass them "
        "each call."))
    candidate: Candidate | None = Field(
        default=None, description="Prospective trade to assess against the book")
    equity: float = Field(default=100_000.0, gt=0, description="Account equity USD")


# --------------------------------------------------------------------------
# Tools: method registry
# --------------------------------------------------------------------------
@mcp.tool(name="quant_list_methods", annotations={"title": "List research methods", **_RO_PURE})
async def quant_list_methods(params: FormatInput) -> str:
    """List every registered paper-backed trading method.

    Args:
        params (FormatInput): response_format ('markdown'|'json').

    Returns:
        str: methods with key, name, paper citation, paper_url and
        regime_applicability. JSON returns the full metadata list; markdown is
        a readable digest. Use this to discover what method_key values are
        valid for quant_analyze_setup / quant_describe_method.
    """
    metas = [m.metadata() for m in methods.list_methods()]
    if params.response_format == ResponseFormat.JSON:
        return to_json({"count": len(metas), "methods": metas})
    lines = [f"# Registered methods ({len(metas)})", ""]
    for m in metas:
        lines += [f"## {m['name']}  (`{m['key']}`)",
                  f"- **Paper**: {m['paper']}",
                  f"- **Link**: {m['paper_url']}",
                  f"- **Regimes**: {m['regime_applicability']}", ""]
    return "\n".join(lines)


@mcp.tool(name="quant_describe_method", annotations={"title": "Describe a method", **_RO_PURE})
async def quant_describe_method(params: DescribeMethodInput) -> str:
    """Full detail and citation for one registered method.

    Args:
        params (DescribeMethodInput): method_key plus response_format.

    Returns:
        str: the method's name, paper, paper_url, regime_applicability and a
        prose description of its mechanics. Error string if the key is unknown
        (call quant_list_methods for valid keys).
    """
    if not methods.has(params.method_key):
        valid = ", ".join(m.key for m in methods.list_methods())
        return _err(f"unknown method_key '{params.method_key}'. Valid: {valid}")
    m = methods.get(params.method_key).metadata()
    if params.response_format == ResponseFormat.JSON:
        return to_json(m)
    return (f"# {m['name']}  (`{m['key']}`)\n\n"
            f"**Paper:** {m['paper']}\n\n**Link:** {m['paper_url']}\n\n"
            f"**Regimes:** {m['regime_applicability']}\n\n{m['description']}")


# --------------------------------------------------------------------------
# Tools: data + indicators
# --------------------------------------------------------------------------
@mcp.tool(name="quant_screen_universe", annotations={"title": "Screen a universe", **_RO_NET})
async def quant_screen_universe(params: ScreenInput) -> str:
    """Rank tickers by intraday tradeability (gap, relative volume, ATR%).

    Hard-filters names under $5 or under $50M average daily dollar volume, then
    scores the rest by 2*|gap%| + rel_volume + 0.5*ATR% and returns the top N.

    Args:
        params (ScreenInput): optional tickers list (defaults to the built-in
        universe), top_n, response_format.

    Returns:
        str: ranked candidates, each with ticker, price, gap_pct, rel_volume,
        atr_pct, avg_dollar_volume_m and score. JSON returns a list under
        'candidates'; markdown is a ranked table.
    """
    universe = [t.upper() for t in params.tickers] if params.tickers else config.DEFAULT_UNIVERSE
    try:
        daily = data.fetch_daily_batch(universe, config.DAILY_LOOKBACK)
    except Exception as e:
        return _err(f"data fetch failed: {type(e).__name__}: {e}")
    if not daily:
        return _err("no daily data returned; check connectivity or symbols")
    results = [r.as_dict() for r in screen(daily, params.top_n)]
    if not results:
        return "No candidates passed the screener filters."
    if params.response_format == ResponseFormat.JSON:
        return to_json({"count": len(results), "candidates": results})
    lines = ["# Screen results", "",
             "| # | Ticker | Price | Gap% | RVol | ATR% | Score |",
             "|---|--------|-------|------|------|------|-------|"]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r['ticker']} | {r['price']} | {r['gap_pct']} | "
                     f"{r['rel_volume']} | {r['atr_pct']} | {r['score']} |")
    return "\n".join(lines)


@mcp.tool(name="quant_compute_indicators", annotations={"title": "Compute indicators", **_RO_NET})
async def quant_compute_indicators(params: TickerInput) -> str:
    """Compute EMA9/EMA20, RSI, ATR, VWAP and the Hurst exponent for a ticker.

    Daily series drive Hurst and ATR; the latest intraday session drives VWAP,
    EMAs and 5m RSI.

    Args:
        params (TickerInput): ticker, response_format.

    Returns:
        str: a dict with price, hurst, atr_daily, and intraday ema9/ema20/
        rsi_5m/vwap. Error string if no data.
    """
    t = params.ticker.upper()
    try:
        daily = data.fetch_daily(t)
        intraday = data.last_session(data.fetch_intraday(t))
    except Exception as e:
        return _err(f"data fetch failed: {type(e).__name__}: {e}")
    if daily.empty or intraday.empty:
        return _err(f"insufficient data for {t}")
    out = {
        "ticker": t,
        "price": round(float(intraday["Close"].iloc[-1]), 2),
        "hurst": round(hurst_exponent(daily["Close"].dropna().iloc[-config.HURST_WINDOW:],
                                      config.HURST_MAX_LAG), 3),
        "atr_daily": round(float(atr(daily).iloc[-1]), 2),
        "ema9": round(float(ema(intraday["Close"], 9).iloc[-1]), 2),
        "ema20": round(float(ema(intraday["Close"], 20).iloc[-1]), 2),
        "rsi_5m": round(float(rsi(intraday["Close"]).iloc[-1]), 1),
        "vwap": round(float(vwap(intraday).iloc[-1]), 2),
    }
    if params.response_format == ResponseFormat.JSON:
        return to_json(out)
    return "\n".join([f"# Indicators — {t}", ""] + [f"- **{k}**: {v}"
                     for k, v in out.items() if k != "ticker"])


@mcp.tool(name="quant_detect_regime", annotations={"title": "Detect market regime", **_RO_NET})
async def quant_detect_regime(params: TickerInput) -> str:
    """Classify a ticker's regime from its daily Hurst exponent.

    H >= 0.55 -> TRENDING (momentum edge), H <= 0.45 -> MEAN_REVERTING (fade
    edge), otherwise RANDOM_WALK (no structural edge).

    Args:
        params (TickerInput): ticker, response_format.

    Returns:
        str: dict with ticker, hurst and regime. Error string if no data.
    """
    t = params.ticker.upper()
    try:
        daily = data.fetch_daily(t)
    except Exception as e:
        return _err(f"data fetch failed: {type(e).__name__}: {e}")
    if daily.empty:
        return _err(f"insufficient data for {t}")
    h = hurst_exponent(daily["Close"].dropna().iloc[-config.HURST_WINDOW:], config.HURST_MAX_LAG)
    regime = ("TRENDING" if h >= config.HURST_TREND
              else "MEAN_REVERTING" if h <= config.HURST_REVERT else "RANDOM_WALK")
    out = {"ticker": t, "hurst": round(h, 3), "regime": regime}
    if params.response_format == ResponseFormat.JSON:
        return to_json(out)
    return f"# Regime — {t}\n\n- **Hurst**: {out['hurst']}\n- **Regime**: {regime}"


# --------------------------------------------------------------------------
# Tools: method analysis + decision
# --------------------------------------------------------------------------
@mcp.tool(name="quant_analyze_setup", annotations={"title": "Analyze a setup", **_RO_NET})
async def quant_analyze_setup(params: AnalyzeInput) -> str:
    """Run a registered method on a ticker to produce a trade setup.

    Fetches daily + latest-session intraday data and dispatches to the named
    method, which returns signal (LONG/SHORT/NO_ENTRY), playbook, regime, and
    entry/stop/target when actionable.

    Args:
        params (AnalyzeInput): ticker, method_key (default 'hurst_regime_orb'),
        response_format.

    Returns:
        str: the setup dict (feed it directly to quant_score_decision). Includes
        signal, playbook, regime, hurst, price, atr_daily, rel_volume, rsi,
        entry, stop, target, reasons, plus method extras (vwap, or_high, or_low,
        ema9, ema20). Error string for unknown method or missing data.
    """
    t = params.ticker.upper()
    if not methods.has(params.method_key):
        valid = ", ".join(m.key for m in methods.list_methods())
        return _err(f"unknown method_key '{params.method_key}'. Valid: {valid}")
    method = methods.get(params.method_key)
    if method.analyze is None:
        return _err(f"'{params.method_key}' is universe-based — use "
                    "quant_analyze_universe instead")
    try:
        daily = data.fetch_daily(t, method.data.daily_period)
        session = None
        ctx: dict = {"ticker": t}
        if method.data.needs_intraday:
            intraday = data.fetch_intraday(t)
            session = data.last_session(intraday)
            if session.empty:
                return _err(f"no intraday data for {t}")
            vol20 = daily["Volume"].iloc[-21:-1].mean()
            ctx["rel_volume"] = prorated_rel_volume(intraday, float(vol20))
    except Exception as e:
        return _err(f"data fetch failed: {type(e).__name__}: {e}")
    if daily.empty:
        return _err(f"insufficient data for {t}")
    setup = method.analyze(daily, session, ctx)
    sd = setup.as_dict()
    # higher-timeframe context for the decision helper's optional 6th factor
    closes = daily["Close"].dropna()
    sd["htf_alignment"] = {
        "price": round(float(closes.iloc[-1]), 2),
        "ema20": round(float(ema(closes, 20).iloc[-1]), 2),
        "ema50": round(float(ema(closes, 50).iloc[-1]), 2),
    }
    if params.response_format == ResponseFormat.JSON:
        return to_json(sd)
    lines = [f"# Setup — {t}  ({method.name})", "",
             f"- **Signal**: {sd['signal']}",
             f"- **Playbook / Regime**: {sd['playbook']} / {sd['regime']} (H={sd['hurst']})"]
    if sd["signal"] != "NO_ENTRY":
        lines += [f"- **Entry / Stop / Target**: {sd['entry']} / {sd['stop']} / {sd['target']}"]
    lines += ["- **Reasons**: " + "; ".join(sd["reasons"])]
    return "\n".join(lines)


@mcp.tool(name="quant_score_decision", annotations={"title": "Score an entry decision", **_RO_PURE})
async def quant_score_decision(params: ScoreDecisionInput) -> str:
    """Deterministically score a setup and decide ENTRY / NO_ENTRY.

    This is the decision helper: a pure function (no LLM, no network). It
    combines scientifically-motivated sub-factors — reward:risk, regime
    strength (|Hurst-0.5|), volume confirmation, RSI positioning and
    ATR-normalised stop quality — into a 0..1 composite score, gates it against
    a threshold, applies an optional caller-supplied news-sentiment
    boost/veto, and returns fixed-fractional position sizing plus a mechanical
    execution plan. Same inputs always yield the same output.

    Args:
        params (ScoreDecisionInput): setup (from quant_analyze_setup), equity,
        risk_pct, optional news_sentiment ('bullish'|'bearish'|'neutral') and
        news_confidence (0..1). The AGENT derives the sentiment from headlines;
        this tool only consumes it.

    Returns:
        str: dict with verdict (ENTRY|NO_ENTRY), score, threshold, per-factor
        breakdown, factor_weights, news boost/veto, rationale, position_size
        (shares, dollar_risk, formula, haircut flag) and, on ENTRY, an
        execution_plan (entry trigger, order type, stop ladder, profit taking,
        time stop, abort conditions). 'deterministic': true.
    """
    try:
        result = decision.score_decision(
            params.setup, equity=params.equity, risk_pct=params.risk_pct,
            news_sentiment=params.news_sentiment,
            news_confidence=params.news_confidence,
            days_to_earnings=params.days_to_earnings,
            portfolio_heat_pct=params.portfolio_heat_pct)
    except Exception as e:
        return _err(f"could not score setup: {type(e).__name__}: {e}. "
                    "Ensure 'setup' came from quant_analyze_setup.")
    if params.response_format == ResponseFormat.JSON:
        return to_json(result)
    lines = [f"# Decision — {result['verdict']}", "",
             f"- **Score**: {result['score']} (threshold {result.get('threshold')})",
             f"- **Rationale**: {result.get('rationale', '')}"]
    if result.get("factors"):
        lines.append("- **Factors**: " + ", ".join(f"{k}={v}" for k, v in result["factors"].items()))
    if result.get("position_size"):
        ps = result["position_size"]
        lines.append(f"- **Size**: {ps['shares']} sh, risk ${ps['dollar_risk']} "
                     f"(haircut={ps['haircut_applied']})")
    plan = result.get("execution_plan")
    if plan:
        lines += ["", "## Execution plan",
                  f"- **Start**: {plan['when_to_start']}",
                  f"- **Trigger**: {plan['entry_trigger']}",
                  f"- **Order**: {plan['order_type']}",
                  f"- **Stop**: {plan['stop_management']}",
                  f"- **Profit**: {plan['profit_taking']}",
                  f"- **Time stop**: {plan['time_stop']}",
                  "- **Abort if**: " + "; ".join(plan["abort_if"])]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Tools: universe analysis, backtesting, events, portfolio, market context
# --------------------------------------------------------------------------
@mcp.tool(name="quant_analyze_universe",
          annotations={"title": "Analyze a universe method", **_RO_NET})
async def quant_analyze_universe(params: AnalyzeUniverseInput) -> str:
    """Run a universe-based method (ranking or pairs) across multiple tickers.

    xs_momentum ranks the universe by 12-1 momentum and marks the top-N book
    LONG. pairs_cointegration requires exactly 2 tickers and returns one setup
    per leg with the shared spread statistics (beta, ADF p-value, z-score).

    Args:
        params (AnalyzeUniverseInput): method_key, tickers (defaults to the
        built-in universe), top_n (ranking methods), response_format.

    Returns:
        str: list of setup dicts (each feedable to quant_score_decision).
        Error string for unknown/non-universe methods or bad ticker counts.
    """
    if not methods.has(params.method_key):
        valid = ", ".join(m.key for m in methods.list_methods())
        return _err(f"unknown method_key '{params.method_key}'. Valid: {valid}")
    method = methods.get(params.method_key)
    if method.analyze_universe is None:
        return _err(f"'{params.method_key}' is per-ticker — use quant_analyze_setup")
    tickers = ([t.upper() for t in params.tickers] if params.tickers
               else config.DEFAULT_UNIVERSE)
    lo, hi = method.data.min_tickers, method.data.max_tickers
    if not (lo <= len(tickers) <= hi):
        return _err(f"'{params.method_key}' needs between {lo} and {hi} tickers, "
                    f"got {len(tickers)}")
    try:
        daily_map = data.fetch_daily_batch(tickers, method.data.daily_period)
        setups = method.analyze_universe(daily_map, {"top_n": params.top_n})
    except Exception as e:
        return _err(f"analysis failed: {type(e).__name__}: {e}")
    dicts = [s.as_dict() for s in setups]
    if params.response_format == ResponseFormat.JSON:
        return to_json({"method": params.method_key, "count": len(dicts),
                        "setups": dicts})
    lines = [f"# {method.name} — {len(dicts)} setups", ""]
    for d in dicts:
        sig = d["signal"]
        mark = "**" if sig != "NO_ENTRY" else ""
        lines.append(f"- {mark}{d['ticker']}: {sig}{mark} — "
                     + "; ".join(d.get("reasons", [])))
    return "\n".join(lines)


@mcp.tool(name="quant_backtest_method",
          annotations={"title": "Backtest a method", **_RO_NET})
async def quant_backtest_method(params: BacktestInput) -> str:
    """Backtest a registered method by replaying its own analyze() over history.

    The engines execute the method's live signal logic bar by bar (no separate
    backtest implementation that could drift). Daily methods replay daily bars
    with next-open fills and method-specific exits; intraday methods replay
    each available 5m session (provider-capped to ~60 days); xs_momentum runs
    a monthly-rebalance portfolio. Costs applied one-way on entry and exit.

    Args:
        params (BacktestInput): method_key, tickers, period, costs_bps,
        risk_pct, top_n, response_format.

    Returns:
        str: stats (n_trades, win_rate, avg_r/expectancy, profit_factor,
        max_drawdown_pct, total_return_pct), in-sample/out-of-sample halves,
        and the last 10 trades. Treat results as PARAMETER VALIDATION, not a
        forecast — yfinance data is survivorship-prone and costs are estimates.
    """
    if not methods.has(params.method_key):
        valid = ", ".join(m.key for m in methods.list_methods())
        return _err(f"unknown method_key '{params.method_key}'. Valid: {valid}")
    method = methods.get(params.method_key)
    tickers = [t.upper() for t in params.tickers]
    try:
        period = method.data.daily_period if params.period == "2y" else params.period
        daily_map = data.fetch_daily_batch(tickers, max(period, "2y", key=len))
        intraday_map = {}
        if method.timeframe == "intraday":
            intraday_map = {t: data.fetch_intraday(t, period="1mo") for t in tickers}
        result = bt.run_backtest(params.method_key, tickers, daily_map,
                                 intraday_map, params.costs_bps,
                                 params.risk_pct, params.top_n)
    except Exception as e:
        return _err(f"backtest failed: {type(e).__name__}: {e}")
    if params.response_format == ResponseFormat.JSON:
        return to_json(result)
    lines = [f"# Backtest — {params.method_key} on {', '.join(tickers)}", ""]
    stats = result.get("stats") or result
    for k, v in stats.items():
        if not isinstance(v, (list, dict)):
            lines.append(f"- **{k}**: {v}")
    if result.get("is_oos"):
        lines.append("")
        lines.append("## Stability (first half vs second half)")
        for name, s in result["is_oos"].items():
            lines.append(f"- {name}: n={s.get('n_trades')}, "
                         f"win_rate={s.get('win_rate')}, avg_r={s.get('avg_r')}")
    return "\n".join(lines)


@mcp.tool(name="quant_check_events",
          annotations={"title": "Check earnings/dividend events", **_RO_NET})
async def quant_check_events(params: TickerInput) -> str:
    """Upcoming earnings and dividend events for a ticker.

    Earnings inside the holding window are the classic binary risk: pass
    days_to_earnings into quant_score_decision, which vetoes swing/position
    entries within the configured window (default 3 days) and warns intraday.

    Args:
        params (TickerInput): ticker, response_format.

    Returns:
        str: {next_earnings, days_to_earnings, recent_earnings, ex_dividend}.
        Fields are null when the provider has no data — treat unknown as risk
        and check the company's IR page before a swing entry.
    """
    t = params.ticker.upper()
    try:
        ev = data.get_events(t)
    except Exception as e:
        return _err(f"event fetch failed: {type(e).__name__}: {e}")
    if params.response_format == ResponseFormat.JSON:
        return to_json({"ticker": t, **ev})
    lines = [f"# Events — {t}", "",
             f"- **Next earnings**: {ev['next_earnings'] or 'unknown'}"
             + (f" ({ev['days_to_earnings']} days)" if ev['days_to_earnings']
                is not None else ""),
             f"- **Ex-dividend**: {ev['ex_dividend'] or 'unknown'}"]
    if ev["recent_earnings"]:
        lines.append(f"- **Recent earnings**: {', '.join(ev['recent_earnings'])}")
    return "\n".join(lines)


@mcp.tool(name="quant_portfolio_risk",
          annotations={"title": "Assess portfolio risk", **_RO_NET})
async def quant_portfolio_risk(params: PortfolioRiskInput) -> str:
    """Stateless portfolio heat / correlation / concentration check.

    Pass your open positions (the server stores nothing) and optionally a
    candidate trade. Returns per-position and total open risk vs the heat
    limit, pairwise 90d correlation flags, sector concentration, and for the
    candidate a FITS / REDUCE / REJECT verdict with a recommended risk_pct to
    feed into quant_score_decision (with portfolio_heat_pct).

    Args:
        params (PortfolioRiskInput): positions [{ticker, direction, entry,
        stop, shares}], optional candidate {ticker, direction, entry, stop},
        equity, response_format.

    Returns:
        str: heat/correlation/concentration analysis dict; candidate verdict
        when one was supplied.
    """
    try:
        result = portfolio.assess([p.model_dump() for p in params.positions],
                                  params.candidate.model_dump()
                                  if params.candidate else None,
                                  params.equity)
    except Exception as e:
        return _err(f"portfolio assessment failed: {type(e).__name__}: {e}")
    if params.response_format == ResponseFormat.JSON:
        return to_json(result)
    lines = [f"# Portfolio risk — {result['open_positions']} positions, "
             f"heat {result['portfolio_heat_pct']}% / max {result['max_heat_pct']}%",
             ""]
    for p in result["positions"]:
        lines.append(f"- {p['ticker']} {p['direction']}: risk "
                     f"${p['open_risk_usd']} ({p['open_risk_pct']}%)")
    for f in result["correlation_flags"]:
        lines.append(f"- correlation flag: {f['pair'][0]}/{f['pair'][1]} = "
                     f"{f['correlation']}")
    for f in result["sector_concentration_flags"]:
        lines.append(f"- sector concentration: {f['sector']} x{f['count']} "
                     f"({', '.join(f['tickers'])})")
    cand = result.get("candidate")
    if cand:
        lines += ["", f"## Candidate {cand['ticker']} {cand['direction']}: "
                      f"**{cand['verdict']}**",
                  f"- recommended risk: {cand['recommended_risk_pct']}% "
                  f"(avg corr to book: {cand['avg_correlation_to_book']})",
                  "- " + "; ".join(cand["reasons"])]
    return "\n".join(lines)


@mcp.tool(name="quant_market_context",
          annotations={"title": "Market tape context", **_RO_NET})
async def quant_market_context(params: FormatInput) -> str:
    """Index/VIX/sector tape read for the current session.

    SPY/QQQ/IWM day % and 20d-EMA side, VIX level and change, the 11 SPDR
    sector ETFs ranked by day %, and a breadth flag (risk_on when most sectors
    advance). Use before sizing any intraday trade; trading breakouts against
    a risk-off tape is the most common ORB failure mode.

    Args:
        params (FormatInput): response_format.

    Returns:
        str: {indexes, vix, sectors (ranked), breadth{risk_on}}.
    """
    try:
        ctx = market.context()
    except Exception as e:
        return _err(f"market context failed: {type(e).__name__}: {e}")
    if params.response_format == ResponseFormat.JSON:
        return to_json(ctx)
    lines = ["# Market context", ""]
    for t, s in ctx["indexes"].items():
        lines.append(f"- **{t}**: {s['last']} ({s['day_pct']:+.2f}%), "
                     f"{'above' if s['above_20d_ema'] else 'below'} 20d EMA")
    if ctx.get("vix"):
        lines.append(f"- **VIX**: {ctx['vix']['last']} "
                     f"({ctx['vix']['day_change']:+.2f})")
    b = ctx["breadth"]
    lines.append(f"- **Breadth**: {b['sectors_advancing']}/{b['sectors_total']} "
                 f"sectors up — {'risk-ON' if b['risk_on'] else 'risk-OFF'}")
    lines.append("")
    lines.append("| Sector | ETF | Day % |")
    lines.append("|---|---|---|")
    for s in ctx["sectors"]:
        lines.append(f"| {s['sector']} | {s['etf']} | {s['day_pct']:+.2f}% |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Tools: news (data only)
# --------------------------------------------------------------------------
@mcp.tool(name="quant_fetch_news", annotations={"title": "Fetch ticker headlines", **_RO_NET})
async def quant_fetch_news(params: TickerInput) -> str:
    """Fetch recent headlines for a ticker. Data only — no sentiment is computed.

    The calling agent reads these and forms its own sentiment to pass to
    quant_score_decision; this tool deliberately does no interpretation.

    Args:
        params (TickerInput): ticker, response_format.

    Returns:
        str: list of headlines, each with title, summary, published, provider.
        Empty-list message if none found.
    """
    t = params.ticker.upper()
    try:
        items = data.get_headlines(t, limit=config.__dict__.get("NEWS_LIMIT", 8))
    except Exception as e:
        return _err(f"news fetch failed: {type(e).__name__}: {e}")
    if not items:
        return f"No recent headlines found for {t}."
    if params.response_format == ResponseFormat.JSON:
        return to_json({"ticker": t, "count": len(items), "headlines": items})
    lines = [f"# Headlines — {t}", ""]
    for h in items:
        lines.append(f"- **{h['title']}** ({h['provider']}, {h['published']})")
        if h["summary"]:
            lines.append(f"  {h['summary']}")
    return "\n".join(lines)


def main() -> None:
    """Console-script entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
