"""Baselines, a global LightGBM model on log sales, rolling-origin backtesting, and conformal intervals."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURES, align_categories, build_features, fit_store_stats

LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


# ------------------------------------------------------------------------------------------- metrics
def rmspe(y, yhat) -> float:
    """Root mean squared percentage error over days with sales > 0 (the competition metric)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y > 0
    return float(np.sqrt(np.mean(((y[m] - yhat[m]) / y[m]) ** 2)))


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum())


def metrics(y, yhat) -> dict:
    return {"rmspe": rmspe(y, yhat), "wape": wape(y, yhat), "mae": float(np.mean(np.abs(np.asarray(y) - yhat)))}


# ------------------------------------------------------------------------------------------- baselines
def baseline_last_weeks(history, future, weeks: int = 6):
    """Mean of the same weekday over the last `weeks` weeks before the cutoff (per store)."""
    h = history[(history.Open == 1) & (history.Date > history.Date.max() - pd.Timedelta(weeks=weeks))]
    m = h.groupby(["Store", "DayOfWeek"]).Sales.mean().rename("pred").reset_index()
    out = future[["Store", "DayOfWeek"]].merge(m, on=["Store", "DayOfWeek"], how="left").pred.fillna(0).values
    return np.where(future.Open.values == 1, out, 0)


def baseline_store_dow_promo(history, future, days: int = 365):
    """Median sales per store x weekday x promo over the last year."""
    h = history[(history.Open == 1) & (history.Sales > 0) & (history.Date > history.Date.max() - pd.Timedelta(days=days))]
    m = h.groupby(["Store", "DayOfWeek", "Promo"]).Sales.median().rename("pred").reset_index()
    out = future[["Store", "DayOfWeek", "Promo"]].merge(m, on=["Store", "DayOfWeek", "Promo"], how="left").pred
    fallback = future[["Store"]].merge(h.groupby("Store").Sales.median().rename("f").reset_index(), on="Store", how="left").f
    return np.where(future.Open.values == 1, out.fillna(fallback).fillna(0).values, 0)


# ------------------------------------------------------------------------------------------- LightGBM
@dataclass
class ForecastModel:
    booster: lgb.Booster
    stats: pd.DataFrame
    categories: dict
    features: list

    def predict(self, rows: pd.DataFrame, store: pd.DataFrame) -> np.ndarray:
        F = build_features(rows, store, self.stats)
        X = align_categories(F[self.features], self.categories)
        pred = np.expm1(self.booster.predict(X))
        return np.where(F.Open.values == 1, np.clip(pred, 0, None), 0.0)


def fit_lgbm(history, store, features=FEATURES, num_boost_round=1500, params=None, valid=None):
    """Train on open days with sales > 0, target log1p(sales). `valid` (rows with Sales) enables early stopping."""
    stats = fit_store_stats(history)
    tr = history[(history.Open == 1) & (history.Sales > 0)]
    F = build_features(tr, store, stats)
    cats = {c: [str(x) for x in F[c].cat.categories] for c in CATEGORICAL if c in features}
    X = align_categories(F[features], cats)
    dtrain = lgb.Dataset(X, np.log1p(tr.Sales.values), categorical_feature=[c for c in CATEGORICAL if c in features])
    p = {**LGB_PARAMS, **(params or {})}
    if valid is not None:
        v = valid[(valid.Open == 1) & (valid.Sales > 0)]
        Xv = align_categories(build_features(v, store, stats)[features], cats)
        dval = lgb.Dataset(Xv, np.log1p(v.Sales.values), reference=dtrain)
        booster = lgb.train(p, dtrain, num_boost_round, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        booster = lgb.train(p, dtrain, num_boost_round)
    return ForecastModel(booster, stats, cats, list(features))


# ------------------------------------------------------------------------------------------- backtesting
def make_folds(train: pd.DataFrame, n_folds: int = 3, horizon: int = 48):
    """Rolling-origin folds: each validation window is the `horizon` days right after its cutoff, walking backwards
    from the end of the data (mimics the 48-day Kaggle test horizon)."""
    end = train.Date.max()
    folds = []
    for k in range(n_folds):
        v_end = end - pd.Timedelta(days=k * horizon)
        cutoff = v_end - pd.Timedelta(days=horizon)
        folds.append({"fold": k, "cutoff": cutoff, "valid_end": v_end})
    return folds


def split(train, fold):
    hist = train[train.Date <= fold["cutoff"]]
    fut = train[(train.Date > fold["cutoff"]) & (train.Date <= fold["valid_end"])]
    return hist, fut


# ------------------------------------------------------------------------------------------- conformal intervals
def conformal_quantiles(y, yhat, levels=(0.8, 0.95)) -> dict:
    """Split conformal on the log scale: q_alpha of |log1p(y) - log1p(yhat)| on a calibration fold (open, sales>0)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = (y > 0) & (yhat > 0)
    r = np.abs(np.log1p(y[m]) - np.log1p(yhat[m]))
    n = len(r)
    return {lv: float(np.quantile(r, min(1.0, np.ceil((n + 1) * lv) / n))) for lv in levels}


def interval(yhat, q):
    yhat = np.asarray(yhat, float)
    return np.expm1(np.log1p(yhat) - q).clip(min=0), np.expm1(np.log1p(yhat) + q)


def coverage(y, lo, hi, mask=None) -> float:
    y = np.asarray(y, float)
    m = (y > 0) if mask is None else mask
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


# ------------------------------------------------------------------------------------------- monitoring
def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index between a reference and a current sample of one numeric feature.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    e, a = pd.Series(expected).dropna().values, pd.Series(actual).dropna().values
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


# ------------------------------------------------------------------------------------------- export / load (serving)
def save_model(model: ForecastModel, store: pd.DataFrame, conformal: dict, meta: dict, out_dir) -> None:
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(out / "model.txt"))
    model.stats.to_csv(out / "store_stats.csv", index=False)
    store.to_csv(out / "store.csv", index=False)
    json.dump({**meta, "features": model.features, "categories": model.categories,
               "conformal_log_q": {str(k): v for k, v in conformal.items()}}, open(out / "meta.json", "w"), indent=2,
              default=str)


def load_model(model_dir):
    import json
    from pathlib import Path

    d = Path(model_dir)
    meta = json.load(open(d / "meta.json"))
    model = ForecastModel(lgb.Booster(model_file=str(d / "model.txt")), pd.read_csv(d / "store_stats.csv"),
                          meta["categories"], meta["features"])
    store = pd.read_csv(d / "store.csv")
    return model, store, meta
