"""Portfolio risk math with the data layer monkeypatched (no network)."""

import numpy as np
import pytest

from quant_research_mcp import portfolio
from tests.conftest import make_daily


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    rng = np.random.default_rng(9)
    base = rng.normal(0, 1, 200)

    def fake_batch(tickers, period):
        out = {}
        for t in sorted(tickers):
            # AAA and BBB are near-perfectly correlated; everything else
            # is an independent walk
            if t == "AAA":
                series = base
            elif t == "BBB":
                series = base + rng.normal(0, 0.1, 200)
            else:
                series = rng.normal(0, 1, 200)
            out[t] = make_daily(100 + np.cumsum(series))
        return out

    monkeypatch.setattr(portfolio.data, "fetch_daily_batch", fake_batch)
    monkeypatch.setattr(portfolio.data, "get_sector",
                        lambda t: "Technology" if t in ("AAA", "BBB", "CCC")
                        else "Energy")


P1 = {"ticker": "AAA", "direction": "LONG", "entry": 100.0, "stop": 95.0,
      "shares": 100}   # $500 risk
P2 = {"ticker": "BBB", "direction": "LONG", "entry": 50.0, "stop": 48.0,
      "shares": 200}   # $400 risk


def test_heat_math_exact():
    r = portfolio.assess([P1, P2], None, equity=100_000)
    assert r["positions"][0]["open_risk_usd"] == 500.0
    assert r["positions"][1]["open_risk_usd"] == 400.0
    assert r["portfolio_heat_pct"] == 0.9          # 900 / 100k
    assert r["headroom_pct"] == 1.1                # 2% cap - 0.9%


def test_correlation_flagged():
    r = portfolio.assess([P1, P2], None, equity=100_000)
    assert any(set(f["pair"]) == {"AAA", "BBB"} for f in r["correlation_flags"])


def test_candidate_fits_with_headroom():
    cand = {"ticker": "DDD", "direction": "LONG", "entry": 80.0, "stop": 76.0}
    r = portfolio.assess([P1], cand, equity=100_000)
    assert r["candidate"]["verdict"] == "FITS"
    assert r["candidate"]["recommended_risk_pct"] == 0.5


def test_candidate_rejected_at_heat_limit():
    # one big position: $2,000 risk on 100k = at the 2% cap
    big = {"ticker": "AAA", "direction": "LONG", "entry": 100.0, "stop": 80.0,
           "shares": 100}
    cand = {"ticker": "DDD", "direction": "LONG", "entry": 80.0, "stop": 76.0}
    r = portfolio.assess([big], cand, equity=100_000)
    assert r["candidate"]["verdict"] == "REJECT"


def test_candidate_reduced_on_correlation():
    # candidate BBB correlates ~1.0 with open AAA -> REDUCE + halved risk
    cand = {"ticker": "BBB", "direction": "LONG", "entry": 50.0, "stop": 48.0}
    r = portfolio.assess([P1], cand, equity=100_000)
    assert r["candidate"]["verdict"] == "REDUCE"
    assert r["candidate"]["recommended_risk_pct"] == 0.25


def test_sector_concentration_flag():
    p3 = {"ticker": "CCC", "direction": "LONG", "entry": 30.0, "stop": 29.0,
          "shares": 100}
    r = portfolio.assess([P1, P2, p3], None, equity=100_000)
    assert any(f["sector"] == "Technology" and f["count"] == 3
               for f in r["sector_concentration_flags"])
