import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecast import models as M  # noqa: E402
from forecast.data import make_synthetic  # noqa: E402
from forecast.features import build_features, fit_store_stats  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return make_synthetic(12, seed=1)


def test_rmspe_ignores_zero_days():
    assert M.rmspe([100, 0, 200], [110, 50, 180]) == pytest.approx(np.sqrt((0.01 + 0.01) / 2))


def test_store_stats_use_only_history(data):
    train, _, _ = data
    cutoff = pd.Timestamp("2015-01-31")
    hist = train[train.Date <= cutoff]
    s1 = fit_store_stats(hist)
    poisoned = train.copy()
    poisoned.loc[poisoned.Date > cutoff, "Sales"] *= 10      # future changes must not affect stats fit on history
    s2 = fit_store_stats(poisoned[poisoned.Date <= cutoff])
    pd.testing.assert_frame_equal(s1, s2)


def test_feature_row_order_and_single_row(data):
    train, test, store = data
    stats = fit_store_stats(train)
    sample = test.sample(50, random_state=0)
    F = build_features(sample, store, stats)
    assert (F.index == sample.index).all() and (F.Store.values == sample.Store.values).all()
    one = build_features(test.iloc[[3]], store, stats)          # API may send a single row
    assert len(one) == 1


def test_model_beats_baseline_and_conformal_covers(data):
    train, _, store = data
    folds = M.make_folds(train, 2)
    cal_hist, cal_fut = M.split(train, folds[1])
    ev_hist, ev_fut = M.split(train, folds[0])
    m_cal = M.fit_lgbm(cal_hist, store, num_boost_round=200)
    q = M.conformal_quantiles(cal_fut.Sales, m_cal.predict(cal_fut, store))
    m = M.fit_lgbm(ev_hist, store, num_boost_round=200)
    p = m.predict(ev_fut, store)
    assert M.rmspe(ev_fut.Sales, p) < M.rmspe(ev_fut.Sales, M.baseline_last_weeks(ev_hist, ev_fut))
    lo, hi = M.interval(p, q[0.8])
    assert 0.7 < M.coverage(ev_fut.Sales.values, lo, hi) < 0.95
    assert (p[ev_fut.Open.values == 0] == 0).all()


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert M.psi(a, rng.normal(size=5000)) < 0.05
    assert M.psi(a, rng.normal(1.0, 1, 5000)) > 0.25
