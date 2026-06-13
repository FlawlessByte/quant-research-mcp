"""Deterministic decision helper (no LLM, no network, no subprocess).

Given a method's setup dict, account parameters and an OPTIONAL caller-supplied
news-sentiment signal, compute a reproducible composite quality score, an
ENTRY / NO_ENTRY verdict, a fixed-fractional position size and a mechanically
derived execution plan.

The "scientific method" guarantee: every output is a pure function of the
inputs. The same inputs always produce the same outputs (verifiable by calling
twice). The MCP supplies math; the calling agent supplies judgement (it is the
agent that interprets news into the sentiment signal this module consumes).
"""


from . import config


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_reward_risk(entry: float, stop: float, target: float) -> float:
    """Realised reward:risk vs the configured target multiple, scaled to 0..1."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0
    rr = reward / risk
    return _clip01(rr / config.RISK_REWARD)        # 1.0 when rr meets the target


def _score_regime_strength(hurst: float | None) -> float:
    """Distance of Hurst from the 0.5 random walk, scaled. |H-0.5|=0.1 -> ~1.0."""
    if hurst is None:
        return 0.5
    return _clip01(abs(hurst - 0.5) / 0.10)


def _score_volume(rel_volume: float) -> float:
    """Relative volume above the floor. At floor -> 0.5, at 2x floor -> 1.0."""
    floor = config.MIN_REL_VOLUME
    if rel_volume <= 0:
        return 0.0
    return _clip01(0.5 * rel_volume / floor)


def _score_momentum(signal: str, rsi: float, playbook: str = "") -> float:
    """RSI positioning appropriate to the direction AND the playbook.

    The ideal RSI is inverted between momentum and mean-reversion playbooks:

    - Momentum (TREND_ORB / MIXED breakouts): a long wants a healthy 50-70 band
      (momentum, not exhausted) so ideal 60; a short wants 30-50 so ideal 40.
    - Mean reversion (REVERT_VWAP_FADE): the edge IS the extreme. A fade short
      wants overbought RSI (>=70) so ideal 75; a fade long wants oversold RSI
      (<=30) so ideal 25. Scoring these as momentum setups would penalise the
      very RSI extreme the fade depends on.

    Score peaks at the ideal and falls off linearly over a 30-point spread.
    """
    is_fade = "FADE" in playbook.upper() or "REVERT" in playbook.upper()
    if signal == "LONG":
        ideal = 25.0 if is_fade else 60.0
    elif signal == "SHORT":
        ideal = 75.0 if is_fade else 40.0
    else:
        return 0.0
    return _clip01(1.0 - abs(rsi - ideal) / 30.0)


def _score_stop_quality(entry: float, stop: float, atr_daily: float) -> float:
    """Stop distance normalised by daily ATR.

    Too tight (< NOISE_STOP_ATR_FRACTION x ATR) -> high noise-stop risk, low
    score. Comfortable around 0.5x ATR. Beyond ~1.2x ATR the trade risks too
    much per share, so the score tapers again.
    """
    if atr_daily <= 0:
        return 0.5
    frac = abs(entry - stop) / atr_daily
    if frac < config.NOISE_STOP_ATR_FRACTION:
        return _clip01(frac / config.NOISE_STOP_ATR_FRACTION * 0.5)   # 0..0.5
    if frac <= 0.6:
        return 1.0
    return _clip01(1.0 - (frac - 0.6) / 0.6)                          # taper past 0.6x


def _news_factor(sentiment: str | None, confidence: float, signal: str
                 ) -> tuple[float, bool, str]:
    """Map a caller-supplied sentiment into (multiplier, veto, note).

    sentiment in {bullish, bearish, neutral, None}. Aligned news lifts the
    score, contradicting news with conviction vetoes the trade outright.
    """
    if not sentiment or sentiment == "neutral":
        return 1.0, False, "news neutral or not supplied"
    aligns = ((signal == "LONG" and sentiment == "bullish")
              or (signal == "SHORT" and sentiment == "bearish"))
    contradicts = ((signal == "LONG" and sentiment == "bearish")
                   or (signal == "SHORT" and sentiment == "bullish"))
    conf = _clip01(confidence)
    if aligns:
        return 1.0 + 0.15 * conf, False, f"news supports {signal} (conf {conf:.2f})"
    if contradicts:
        if conf >= 0.5:
            return 1.0, True, f"news contradicts {signal} with conviction -> veto"
        return 1.0 - 0.15 * conf, False, f"news mildly against {signal} (conf {conf:.2f})"
    return 1.0, False, "news unrelated to direction"


def _position_size(entry: float, stop: float, atr_daily: float,
                   equity: float, risk_pct: float) -> dict:
    """Fixed-fractional sizing with a haircut for noise-tight stops."""
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0 or equity <= 0:
        return {"shares": 0, "dollar_risk": 0.0, "haircut_applied": False,
                "formula": "n/a (zero risk distance)"}
    haircut = (atr_daily > 0
               and per_share_risk < config.NOISE_STOP_ATR_FRACTION * atr_daily)
    effective_pct = risk_pct * (config.NOISE_STOP_SIZE_HAIRCUT if haircut else 1.0)
    dollar_risk = equity * effective_pct
    shares = int(dollar_risk / per_share_risk)
    return {
        "shares": shares,
        "dollar_risk": round(shares * per_share_risk, 2),
        "risk_pct_used": round(effective_pct * 100, 3),
        "haircut_applied": haircut,
        "formula": (f"shares = floor(equity {equity:.0f} * {effective_pct:.4f} "
                    f"/ per_share_risk {per_share_risk:.2f})"),
    }


def _execution_plan(setup: dict) -> dict:
    """Mechanically derived plan, shaped by the setup's timeframe."""
    signal = setup["signal"]
    timeframe = setup.get("timeframe", "intraday")
    entry, stop, target = setup["entry"], setup["stop"], setup["target"]
    risk = abs(entry - stop)
    long = signal == "LONG"
    r1 = entry + risk if long else entry - risk            # +1R level
    r1_5 = entry + 1.5 * risk if long else entry - 1.5 * risk
    or_high = setup.get("or_high")
    or_low = setup.get("or_low")
    vwap = setup.get("vwap")
    buffer = round(0.05 if entry < 200 else 0.10, 2)

    abort = ["an unscheduled halt or headline hits the name"]
    plan = {
        "initial_stop": round(stop, 2),
        "order_type": (
            f"{'buy' if long else 'sell'} stop-limit: stop {entry:.2f} / "
            f"limit {entry + buffer if long else entry - buffer:.2f}"),
        "abort_if": abort,
    }

    if timeframe == "intraday":
        if vwap is not None:
            side = "below" if long else "above"
            abort.insert(0, f"a 5m candle closes {side} VWAP {vwap:.2f} before entry")
        invalidation = or_low if long else or_high
        if invalidation is not None:
            abort.insert(1, f"price re-enters the opening range past "
                            f"{invalidation:.2f} pre-fill")
        abort.append("relative volume falls below 0.8x mid-trade")
        plan.update({
            "when_to_start": ("act on the live session within the first 90 "
                              "minutes after the open; ORB/fade edge decays "
                              "past late morning"),
            "entry_trigger": (
                f"{'price holds above' if long else 'price holds below'} "
                f"{entry:.2f} on a 5m candle close on the correct VWAP side"),
            "stop_management": (
                f"hard stop {stop:.2f}; move to breakeven {entry:.2f} at +1R "
                f"({r1:.2f}); trail 5m EMA20 beyond +1.5R ({r1_5:.2f})"),
            "profit_taking": (
                f"scale 50% at +1R ({r1:.2f}); hold remainder to target "
                f"{target:.2f} or the EMA20 trail, whichever hits first"),
            "time_stop": ("exit by 15:55 ET regardless; abandon if not +0.5R "
                          "within 45 minutes of entry"),
        })
    elif timeframe == "swing":
        plan.update({
            "when_to_start": ("enter on the next session open (or the close of "
                              "the signal day); signal is computed on daily bars"),
            "entry_trigger": (
                f"{'price at or below' if long else 'price at or above'} "
                f"{entry:.2f} near the open — daily signal, no intraday timing "
                f"needed"),
            "stop_management": (
                f"hard stop {stop:.2f} (daily close basis); move to breakeven "
                f"{entry:.2f} after a daily close beyond +1R ({r1:.2f})"),
            "profit_taking": (
                f"primary exit per the method (e.g. 5d-SMA cross / z-score 0); "
                f"target reference {target:.2f}; scale 50% at +1R ({r1:.2f}) "
                f"if reached first"),
            "time_stop": ("re-evaluate after 10 trading days; exit if the "
                          "setup thesis hasn't begun to work"),
        })
        abort.append("a daily close beyond the stop level")
        abort.append("earnings inside the expected holding window (re-check "
                     "quant_check_events)")
    else:  # position
        plan.update({
            "when_to_start": ("enter on the next session open; breakout is "
                              "computed on daily closes — intraday timing is "
                              "noise at this horizon"),
            "entry_trigger": (
                f"{'price above' if long else 'price below'} {entry:.2f} at "
                f"or after the next open"),
            "stop_management": (
                f"initial stop {stop:.2f}; trail per the method (20d opposite "
                f"channel / 2.5x ATR), updated on each daily close — never "
                f"widened"),
            "profit_taking": (
                "no fixed target — trend methods ride winners; the trail is "
                f"the exit (reference 2R = {target:.2f})"),
            "time_stop": ("weekly review; exit only on trail/channel breach, "
                          "not on time"),
        })
        abort.append("a daily close beyond the stop level")

    return plan


def _score_htf_alignment(signal: str, htf: dict) -> float:
    """Higher-timeframe confirmation: daily price vs 20d/50d EMA alignment.

    htf = {price, ema20, ema50}. Full score when both EMAs are on the trade's
    side and stacked; half when mixed; zero when fully against.
    """
    price, e20, e50 = htf.get("price"), htf.get("ema20"), htf.get("ema50")
    if price is None or e20 is None or e50 is None:
        return 0.5
    if signal == "LONG":
        checks = [price > e20, price > e50, e20 > e50]
    else:
        checks = [price < e20, price < e50, e20 < e50]
    return sum(checks) / 3.0


def score_decision(setup: dict, equity: float = 100_000.0,
                   risk_pct: float = config.DEFAULT_ACCOUNT_RISK_PCT,
                   news_sentiment: str | None = None,
                   news_confidence: float = 0.0,
                   days_to_earnings: int | None = None,
                   portfolio_heat_pct: float | None = None) -> dict:
    """Score a setup and decide ENTRY / NO_ENTRY deterministically.

    Optional risk overlays (all caller-supplied; the function stays pure):
    - days_to_earnings: vetoes swing/position entries inside
      EARNINGS_VETO_DAYS; intraday setups get a warning flag only.
    - portfolio_heat_pct: current open risk as % of equity (e.g. from
      quant_portfolio_risk); vetoes when already at/above MAX_PORTFOLIO_HEAT.
    - setup["htf_alignment"] = {price, ema20, ema50}: when present, adds a
      sixth higher-timeframe confirmation factor; when absent the legacy
      5-factor weights apply unchanged.

    Returns score, per-factor breakdown, verdict, position size and (on ENTRY)
    a timeframe-appropriate execution plan. Pure function of the inputs.
    """
    signal = setup.get("signal", "NO_ENTRY")
    if signal == "NO_ENTRY" or setup.get("entry") is None:
        return {
            "verdict": "NO_ENTRY",
            "score": 0.0,
            "reason": "method produced no actionable signal",
            "factors": {},
            "position_size": None,
            "execution_plan": None,
        }

    timeframe = setup.get("timeframe", "intraday")
    entry, stop, target = setup["entry"], setup["stop"], setup["target"]
    factors = {
        "reward_risk": _score_reward_risk(entry, stop, target),
        "regime_strength": _score_regime_strength(setup.get("hurst")),
        "volume_confirmation": _score_volume(setup.get("rel_volume", 0.0)),
        "momentum_position": _score_momentum(signal, setup.get("rsi", 50.0),
                                              setup.get("playbook", "")),
        "stop_quality": _score_stop_quality(entry, stop, setup.get("atr_daily", 0.0)),
    }
    weights = dict(config.DECISION_WEIGHTS)
    htf = setup.get("htf_alignment")
    if isinstance(htf, dict):
        # Blend in the 6th factor at 15%, shrinking the others proportionally.
        factors["htf_alignment"] = _score_htf_alignment(signal, htf)
        weights = {k: v * 0.85 for k, v in weights.items()}
        weights["htf_alignment"] = 0.15

    base = sum(weights[k] * v for k, v in factors.items())
    news_mult, news_veto, news_note = _news_factor(news_sentiment,
                                                   news_confidence, signal)
    score = _clip01(base * news_mult)

    vetoes: list[str] = []
    warnings: list[str] = []
    if news_veto:
        vetoes.append("news contradicts the signal with conviction")
    if days_to_earnings is not None and 0 <= days_to_earnings <= config.EARNINGS_VETO_DAYS:
        msg = (f"earnings in {days_to_earnings} day(s) — inside the "
               f"{config.EARNINGS_VETO_DAYS}-day window")
        if timeframe in ("swing", "position"):
            vetoes.append(msg)
        else:
            warnings.append(msg + " (intraday: manage around the print, "
                                  "do not hold through it)")
    if portfolio_heat_pct is not None and \
            portfolio_heat_pct >= config.MAX_PORTFOLIO_HEAT * 100:
        vetoes.append(
            f"portfolio heat {portfolio_heat_pct:.2f}% at/above the "
            f"{config.MAX_PORTFOLIO_HEAT * 100:.0f}% limit — no new risk")

    verdict = "ENTRY" if (score >= config.DECISION_THRESHOLD and not vetoes) \
        else "NO_ENTRY"
    rationale = [f"composite {score:.2f} vs threshold "
                 f"{config.DECISION_THRESHOLD:.2f}", news_note]
    rationale += [f"VETO: {v}" for v in vetoes]
    rationale += [f"WARNING: {w}" for w in warnings]

    size = _position_size(entry, stop, setup.get("atr_daily", 0.0), equity, risk_pct)
    plan = _execution_plan(setup) if verdict == "ENTRY" else None

    return {
        "verdict": verdict,
        "score": round(score, 4),
        "threshold": config.DECISION_THRESHOLD,
        "direction": signal,
        "timeframe": timeframe,
        "factors": {k: round(v, 4) for k, v in factors.items()},
        "factor_weights": {k: round(v, 4) for k, v in weights.items()},
        "news": {"sentiment": news_sentiment or "none",
                 "confidence": round(_clip01(news_confidence), 2),
                 "multiplier": round(news_mult, 3), "veto": news_veto},
        "vetoes": vetoes,
        "warnings": warnings,
        "rationale": "; ".join(rationale),
        "position_size": size,
        "execution_plan": plan,
        "deterministic": True,
    }
