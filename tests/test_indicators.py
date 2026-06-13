import numpy as np
import pandas as pd

from quant_research_mcp.indicators import (
    atr,
    ema,
    hurst_exponent,
    prorated_rel_volume,
    rsi,
    sma,
    vwap,
)


def test_ema_converges_to_constant():
    s = pd.Series([50.0] * 100)
    assert abs(ema(s, 9).iloc[-1] - 50.0) < 1e-9


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 50))
    dn = pd.Series(np.linspace(200, 100, 50))
    assert rsi(up).iloc[-1] > 90
    assert rsi(dn).iloc[-1] < 10
    assert 0 <= rsi(up).iloc[-1] <= 100


def test_atr_positive_and_scales(trending_daily):
    a = atr(trending_daily)
    assert (a.dropna() > 0).all()


def test_vwap_between_low_and_high(trending_daily):
    v = vwap(trending_daily.iloc[-30:])
    assert (v <= trending_daily["High"].iloc[-30:].max()).all()
    assert (v >= trending_daily["Low"].iloc[-30:].min()).all()


def test_hurst_separates_regimes(trending_daily, mean_reverting_daily):
    h_trend = hurst_exponent(trending_daily["Close"].values)
    h_mr = hurst_exponent(mean_reverting_daily["Close"].values)
    assert h_trend > 0.55
    assert h_mr < 0.45


def test_hurst_short_series_neutral():
    assert hurst_exponent(np.array([1.0, 2.0, 3.0])) == 0.5


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 5).iloc[-1] == 3.0


def test_prorated_rel_volume_mid_session():
    """Today at half the bars with half the volume of a normal day -> ~1.0."""
    idx_prior = pd.date_range("2024-01-02 09:30", periods=78, freq="5min")
    idx_today = pd.date_range("2024-01-03 09:30", periods=39, freq="5min")
    prior = pd.DataFrame({"Volume": np.full(78, 100.0)}, index=idx_prior)
    today = pd.DataFrame({"Volume": np.full(39, 100.0)}, index=idx_today)
    intraday = pd.concat([prior, today])
    vol20 = 7800.0  # matches the prior full-day volume
    rv = prorated_rel_volume(intraday, vol20)
    assert abs(rv - 1.0) < 0.01


def test_prorated_rel_volume_elevated():
    idx_prior = pd.date_range("2024-01-02 09:30", periods=78, freq="5min")
    idx_today = pd.date_range("2024-01-03 09:30", periods=39, freq="5min")
    prior = pd.DataFrame({"Volume": np.full(78, 100.0)}, index=idx_prior)
    today = pd.DataFrame({"Volume": np.full(39, 200.0)}, index=idx_today)  # 2x pace
    rv = prorated_rel_volume(pd.concat([prior, today]), 7800.0)
    assert abs(rv - 2.0) < 0.02
