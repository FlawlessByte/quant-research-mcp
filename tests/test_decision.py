"""Decision helper: eval-locked values, vetoes, htf blend, determinism."""

from quant_research_mcp import decision

BASE = dict(ticker="X", method_key="hurst_regime_orb", signal="LONG",
            playbook="TREND_ORB", regime="TRENDING", hurst=0.60, price=100.0,
            atr_daily=4.0, rel_volume=2.4, rsi=60.0,
            entry=100.0, stop=98.0, target=104.0)


def test_eval_locked_perfect_setup():
    r = decision.score_decision(dict(BASE), equity=100_000, risk_pct=0.005)
    assert r["verdict"] == "ENTRY"
    assert r["score"] == 1.0
    assert r["position_size"]["shares"] == 250


def test_eval_locked_mid_setup():
    s = dict(ticker="X", signal="SHORT", playbook="TREND_ORB", regime="TRENDING",
             hurst=0.53, price=50.0, atr_daily=2.0, rel_volume=1.2, rsi=55.0,
             entry=50.0, stop=51.6, target=46.8)
    r = decision.score_decision(s)
    assert r["score"] == 0.6183
    assert r["factors"]["regime_strength"] == 0.3


def test_eval_locked_fade_momentum_factor():
    s = dict(BASE, signal="SHORT", playbook="REVERT_VWAP_FADE", hurst=0.369,
             price=210.0, atr_daily=8.03, rel_volume=0.59, rsi=74.0,
             entry=210.0, stop=211.0, target=205.0)
    r = decision.score_decision(s)
    assert r["factors"]["momentum_position"] == 0.9667


def test_news_veto_and_boost():
    veto = decision.score_decision(dict(BASE), news_sentiment="bearish",
                                   news_confidence=0.9)
    assert veto["verdict"] == "NO_ENTRY"
    assert veto["news"]["veto"] is True
    boost = decision.score_decision(dict(BASE), news_sentiment="bullish",
                                    news_confidence=1.0)
    assert boost["news"]["multiplier"] == 1.15


def test_noise_stop_haircut():
    s = dict(BASE, stop=99.5, target=101.0)
    r = decision.score_decision(s)
    assert r["position_size"]["haircut_applied"] is True


def test_no_entry_passthrough():
    s = dict(BASE, signal="NO_ENTRY", entry=None)
    assert decision.score_decision(s)["verdict"] == "NO_ENTRY"


def test_deterministic():
    a = decision.score_decision(dict(BASE), news_sentiment="bullish",
                                news_confidence=0.8)
    b = decision.score_decision(dict(BASE), news_sentiment="bullish",
                                news_confidence=0.8)
    assert a == b


def test_earnings_veto_swing_but_warn_intraday():
    swing = dict(BASE, timeframe="swing")
    r = decision.score_decision(swing, days_to_earnings=1)
    assert r["verdict"] == "NO_ENTRY"
    assert any("earnings" in v for v in r["vetoes"])

    intraday = dict(BASE, timeframe="intraday")
    r2 = decision.score_decision(intraday, days_to_earnings=1)
    assert r2["verdict"] == "ENTRY"          # warned, not vetoed
    assert any("earnings" in w for w in r2["warnings"])


def test_earnings_outside_window_no_effect():
    r = decision.score_decision(dict(BASE, timeframe="swing"), days_to_earnings=10)
    assert r["verdict"] == "ENTRY"


def test_portfolio_heat_veto():
    r = decision.score_decision(dict(BASE), portfolio_heat_pct=2.5)
    assert r["verdict"] == "NO_ENTRY"
    r2 = decision.score_decision(dict(BASE), portfolio_heat_pct=1.0)
    assert r2["verdict"] == "ENTRY"


def test_htf_alignment_blend():
    aligned = dict(BASE, htf_alignment={"price": 100, "ema20": 98, "ema50": 96})
    against = dict(BASE, htf_alignment={"price": 100, "ema20": 102, "ema50": 104})
    ra = decision.score_decision(aligned)
    rg = decision.score_decision(against)
    assert ra["factors"]["htf_alignment"] == 1.0
    assert rg["factors"]["htf_alignment"] == 0.0
    assert ra["score"] > rg["score"]
    assert abs(sum(ra["factor_weights"].values()) - 1.0) < 1e-6
    # legacy path untouched when key absent
    legacy = decision.score_decision(dict(BASE))
    assert "htf_alignment" not in legacy["factors"]
    assert legacy["score"] == 1.0


def test_timeframe_execution_plans():
    intraday = decision.score_decision(dict(BASE, timeframe="intraday"))
    swing = decision.score_decision(dict(BASE, timeframe="swing"))
    position = decision.score_decision(dict(BASE, timeframe="position"))
    assert "15:55 ET" in intraday["execution_plan"]["time_stop"]
    assert "10 trading days" in swing["execution_plan"]["time_stop"]
    assert "weekly review" in position["execution_plan"]["time_stop"].lower()
