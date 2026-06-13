"""Shared synthetic-data builders. No network anywhere in the test suite."""

import numpy as np
import pandas as pd
import pytest


def make_daily(closes: np.ndarray, start: str = "2024-01-02",
               vol: float = 1_000_000) -> pd.DataFrame:
    """OHLCV frame around a close series (High/Low bracket the close)."""
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "Open": c.shift(1).fillna(c.iloc[0]),
        "High": c * 1.01,
        "Low": c * 0.99,
        "Close": c,
        "Volume": np.full(len(c), vol),
    })


@pytest.fixture
def trending_daily() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    steps = rng.normal(0.3, 1.0, 500)
    closes = 100 + np.cumsum(np.abs(steps) * 0.5 + 0.2)
    return make_daily(closes)


@pytest.fixture
def mean_reverting_daily() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = np.zeros(500)
    for i in range(1, 500):
        x[i] = -0.6 * x[i - 1] + rng.normal(0, 1)
    return make_daily(100 + x)


@pytest.fixture
def flat_daily() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return make_daily(100 + rng.normal(0, 0.3, 500))
