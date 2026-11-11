"""Feature engineering shared by training, backtesting AND the API (one code path = no train/serve skew).

Forecast horizon is up to 48 days, so no short-term sales lags are used. Store-level history statistics are fit only
on data before the forecast cutoff (`fit_store_stats`) and then joined, which keeps backtests leak-free."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct",
              11: "Nov", 12: "Dec"}
CATEGORICAL = ["StoreType", "Assortment", "StateHoliday"]


def fit_store_stats(history: pd.DataFrame) -> pd.DataFrame:
    """Per-store statistics of log sales on open days, computed on the training window only."""
    h = history[(history.Open == 1) & (history.Sales > 0)].copy()
    h["log_sales"] = np.log1p(h.Sales)
    g = h.groupby("Store")
    stats = pd.DataFrame({
        "store_mean": g.log_sales.mean(), "store_std": g.log_sales.std(),
        "store_cust_mean": np.log1p(g.Customers.mean()) if "Customers" in h else np.nan,
        "store_sales_per_cust": (g.Sales.sum() / g.Customers.sum().clip(lower=1)) if "Customers" in h else np.nan,
    })
    dow = h.pivot_table(index="Store", columns="DayOfWeek", values="log_sales", aggfunc="mean")
    dow.columns = [f"store_dow{c}_mean" for c in dow.columns]
    promo = h.pivot_table(index="Store", columns="Promo", values="log_sales", aggfunc="mean")
    promo.columns = [f"store_promo{c}_mean" for c in promo.columns]
    stats = stats.join(dow).join(promo)
    for c in ["store_promo0_mean", "store_promo1_mean"]:
        if c not in stats:
            stats[c] = np.nan
    stats["store_promo_lift"] = stats["store_promo1_mean"] - stats["store_promo0_mean"]
    # recent level (last 3 months of the window) vs overall: captures trend / refurbishments
    last = h[h.Date > h.Date.max() - pd.Timedelta(days=90)].groupby("Store").log_sales.mean()
    stats["store_recent_mean"] = last
    stats["store_recent_trend"] = stats.store_recent_mean - stats.store_mean
    return stats.reset_index()


def _consecutive_index(flag: pd.Series) -> pd.Series:
    """Position inside a run of consecutive 1s (0 when flag==0)."""
    grp = (flag != flag.shift()).cumsum()
    return flag.groupby(grp).cumsum() * flag


def build_features(df: pd.DataFrame, store: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """df: rows with Store, Date, Open, Promo, StateHoliday, SchoolHoliday (one or many days, one or many stores)."""
    d = df.copy()
    orig_index = d.index
    d = d.reset_index(drop=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d["StateHoliday"] = d["StateHoliday"].astype(str).replace({"0.0": "0", "nan": "0"})
    d = d.merge(store, on="Store", how="left").merge(stats, on="Store", how="left")
    d = d.sort_values(["Store", "Date"]).reset_index(drop=False)

    dt = d.Date.dt
    d["DayOfWeek"] = dt.dayofweek + 1
    d["year"], d["month"], d["day"] = dt.year, dt.month, dt.day
    d["weekofyear"] = dt.isocalendar().week.astype(int).values
    d["dayofyear"] = dt.dayofyear
    d["days_to_month_end"] = dt.days_in_month - dt.day
    d["is_december"] = (d.month == 12).astype(int)
    d["days_to_christmas"] = np.where(d.month == 12, (25 - d.day).clip(lower=-6), 40)

    # competition and Promo2
    comp_open = pd.to_datetime(dict(year=d.CompetitionOpenSinceYear.fillna(1900).astype(int),
                                    month=d.CompetitionOpenSinceMonth.fillna(1).astype(int), day=1), errors="coerce")
    d["comp_months_open"] = ((d.Date - comp_open).dt.days / 30.4).where(d.CompetitionOpenSinceYear.notna()).clip(-12, 200)
    d["log_comp_distance"] = np.log1p(d.CompetitionDistance)
    p2_start = pd.to_datetime(d.Promo2SinceYear.fillna(1900).astype(int).astype(str) + "-01-01", errors="coerce") + \
        pd.to_timedelta((d.Promo2SinceWeek.fillna(1) - 1) * 7, unit="D")
    d["promo2_weeks"] = ((d.Date - p2_start).dt.days / 7).where(d.Promo2 == 1).clip(-10, 300)
    month_abbr = d.month.map(MONTH_ABBR)
    d["promo2_active_month"] = [int(p2 == 1 and isinstance(iv, str) and m in iv.split(",") and w >= 0)
                                for p2, iv, m, w in zip(d.Promo2, d.PromoInterval, month_abbr, d.promo2_weeks.fillna(-1))]

    # neighbouring-day context (known in advance): holidays and promo runs
    by = d.groupby("Store")
    hol = (d.StateHoliday != "0").astype(int)
    d["holiday_tomorrow"] = hol.groupby(d.Store).shift(-1).fillna(0)
    d["holiday_yesterday"] = hol.groupby(d.Store).shift(1).fillna(0)
    d["closed_tomorrow"] = (1 - by.Open.shift(-1).fillna(1))
    d["closed_yesterday"] = (1 - by.Open.shift(1).fillna(1))
    d["promo_day_index"] = d.groupby("Store", group_keys=False).Promo.apply(_consecutive_index)
    d["promo_tomorrow"] = by.Promo.shift(-1).fillna(0)

    # store-history interactions
    dow_cols = [f"store_dow{w}_mean" for w in range(1, 8)]
    for c in dow_cols:
        if c not in d:
            d[c] = np.nan
    d["store_dow_mean"] = d[dow_cols].to_numpy()[np.arange(len(d)), d.DayOfWeek.to_numpy() - 1]
    for c in CATEGORICAL:
        d[c] = d[c].astype("category")
    d = d.set_index("index").sort_index()
    d.index = orig_index            # same row order AND index labels as the input
    return d


FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
    "year", "month", "day", "weekofyear", "dayofyear", "days_to_month_end", "is_december", "days_to_christmas",
    "comp_months_open", "log_comp_distance", "CompetitionDistance", "Promo2", "promo2_weeks", "promo2_active_month",
    "holiday_tomorrow", "holiday_yesterday", "closed_tomorrow", "closed_yesterday", "promo_day_index", "promo_tomorrow",
    "store_mean", "store_std", "store_cust_mean", "store_sales_per_cust", "store_promo_lift", "store_recent_mean",
    "store_recent_trend", "store_dow_mean", "store_promo0_mean", "store_promo1_mean",
]


def align_categories(X: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Use the training categories at prediction time (LightGBM needs identical category codes)."""
    X = X.copy()
    for c, cats in reference.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X
