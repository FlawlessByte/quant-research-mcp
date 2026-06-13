"""Stateless portfolio risk assessment.

The server stores nothing: the caller supplies its open positions and gets
back heat, correlation and concentration analysis plus a recommended max risk
for a candidate trade. Pure math + (cached) price/sector lookups.
"""


import pandas as pd

from . import config, data


def _position_risk(p: dict) -> float:
    """Open dollar risk of one position (entry -> stop, current direction)."""
    return abs(p["entry"] - p["stop"]) * p.get("shares", 0)


def _correlation(tickers: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    if len(tickers) < 2:
        return pd.DataFrame(), []
    daily = data.fetch_daily_batch(tickers, "6mo")
    closes = pd.concat({t: df["Close"] for t, df in daily.items()}, axis=1)
    rets = closes.pct_change().dropna().iloc[-config.CORRELATION_WINDOW:]
    corr = rets.corr()
    flags = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = float(corr.loc[a, b])
            if c >= config.CORRELATION_FLAG:
                flags.append({"pair": [a, b], "correlation": round(c, 2)})
    return corr, flags


def assess(positions: list[dict], candidate: dict | None,
           equity: float) -> dict:
    """positions: [{ticker, direction, entry, stop, shares}]; candidate:
    {ticker, direction, entry, stop} (shares unknown — that's the question)."""
    pos_risks = [{"ticker": p["ticker"],
                  "direction": p.get("direction", "LONG"),
                  "open_risk_usd": round(_position_risk(p), 2),
                  "open_risk_pct": round(_position_risk(p) / equity * 100, 3)}
                 for p in positions]
    total_risk = sum(r["open_risk_usd"] for r in pos_risks)
    heat_pct = total_risk / equity
    headroom_pct = max(0.0, config.MAX_PORTFOLIO_HEAT - heat_pct)

    tickers = [p["ticker"] for p in positions]
    all_tickers = tickers + ([candidate["ticker"]] if candidate else [])
    corr, corr_flags = _correlation(sorted(set(all_tickers)))

    sectors: dict[str, list[str]] = {}
    for t in set(all_tickers):
        s = data.get_sector(t) or "Unknown"
        sectors.setdefault(s, []).append(t)
    sector_flags = [{"sector": s, "tickers": ts, "count": len(ts)}
                    for s, ts in sectors.items() if len(ts) >= 3]

    out = {
        "equity": equity,
        "open_positions": len(positions),
        "max_positions": config.MAX_POSITIONS,
        "positions": pos_risks,
        "portfolio_heat_pct": round(heat_pct * 100, 3),
        "max_heat_pct": config.MAX_PORTFOLIO_HEAT * 100,
        "headroom_pct": round(headroom_pct * 100, 3),
        "correlation_flags": corr_flags,
        "sector_concentration_flags": sector_flags,
    }

    if candidate:
        cand_corr = None
        if not corr.empty and candidate["ticker"] in corr.columns and tickers:
            others = [t for t in tickers if t in corr.columns]
            if others:
                cand_corr = float(corr.loc[candidate["ticker"], others].mean())
        verdict, reasons = "FITS", []
        if len(positions) >= config.MAX_POSITIONS:
            verdict = "REJECT"
            reasons.append(f"already at max positions ({config.MAX_POSITIONS})")
        if headroom_pct <= 0:
            verdict = "REJECT"
            reasons.append(
                f"portfolio heat {heat_pct * 100:.2f}% already at/above the "
                f"{config.MAX_PORTFOLIO_HEAT * 100:.0f}% limit")
        recommended = min(config.DEFAULT_ACCOUNT_RISK_PCT, headroom_pct)
        if verdict != "REJECT" and recommended < config.DEFAULT_ACCOUNT_RISK_PCT:
            verdict = "REDUCE"
            reasons.append(
                f"headroom {headroom_pct * 100:.2f}% below the default "
                f"{config.DEFAULT_ACCOUNT_RISK_PCT * 100:.1f}% per-trade risk")
        if cand_corr is not None and cand_corr >= config.CORRELATION_FLAG:
            if verdict == "FITS":
                verdict = "REDUCE"
            recommended = min(recommended, config.DEFAULT_ACCOUNT_RISK_PCT / 2)
            reasons.append(
                f"candidate's average correlation to the book is "
                f"{cand_corr:.2f} (>= {config.CORRELATION_FLAG}) — halve risk")
        same_dir = [t for p, t in zip(positions, tickers, strict=True)
                    if p.get("direction") == candidate.get("direction")]
        out["candidate"] = {
            "ticker": candidate["ticker"],
            "direction": candidate.get("direction"),
            "verdict": verdict,
            "recommended_risk_pct": round(max(recommended, 0.0) * 100, 3),
            "avg_correlation_to_book": (round(cand_corr, 2)
                                        if cand_corr is not None else None),
            "same_direction_positions": len(same_dir),
            "reasons": reasons or ["within all limits"],
        }
    return out
