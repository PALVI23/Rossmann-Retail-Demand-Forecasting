"""Rossmann Store Sales (Kaggle): 1,115 drug stores, daily sales 2013-01-01..2015-07-31; test = next 48 days."""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITION = "rossmann-store-sales"
FILES = ["train.csv", "test.csv", "store.csv"]


def _secret(name):
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:  # noqa: BLE001
        return os.environ.get(name)


def setup_kaggle():
    token = _secret("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        kd = Path.home() / ".kaggle"
        kd.mkdir(exist_ok=True)
        (kd / "access_token").write_text(token)
        os.chmod(kd / "access_token", 0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.7"], check=True)


def download(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f).exists() for f in FILES):
        return data_dir
    setup_kaggle()
    r = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(data_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{r.stdout}\n{r.stderr}\n"
                           "-> Accept the rules at https://www.kaggle.com/competitions/rossmann-store-sales/rules "
                           "and make sure the Colab secret KAGGLE_API_TOKEN exists with notebook access.")
    for z in data_dir.glob("*.zip"):
        zipfile.ZipFile(z).extractall(data_dir)
    return data_dir


def load(data_dir: str | Path):
    data_dir = Path(data_dir)
    dt = {"StateHoliday": str}
    train = pd.read_csv(data_dir / "train.csv", dtype=dt, parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", dtype=dt, parse_dates=["Date"])
    store = pd.read_csv(data_dir / "store.csv")
    test["Open"] = test["Open"].fillna(1)   # 11 rows of store 622 are missing; assume open (standard choice)
    return train, test, store


def make_synthetic(n_stores: int = 30, seed: int = 0):
    """Rossmann-shaped synthetic data (same columns) for smoke tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", "2015-09-17")
    store = pd.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "StoreType": rng.choice(list("abcd"), n_stores), "Assortment": rng.choice(list("abc"), n_stores),
        "CompetitionDistance": rng.exponential(5000, n_stores).round(),
        "CompetitionOpenSinceMonth": rng.integers(1, 13, n_stores).astype(float),
        "CompetitionOpenSinceYear": rng.integers(2005, 2015, n_stores).astype(float),
        "Promo2": rng.integers(0, 2, n_stores),
    })
    store["Promo2SinceWeek"] = np.where(store.Promo2 == 1, rng.integers(1, 50, n_stores), np.nan)
    store["Promo2SinceYear"] = np.where(store.Promo2 == 1, rng.integers(2010, 2015, n_stores), np.nan)
    store["PromoInterval"] = np.where(store.Promo2 == 1, rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"], n_stores), None)
    rows = []
    for s in store.Store:
        base = rng.uniform(3000, 9000)
        d = pd.DataFrame({"Store": s, "Date": dates})
        d["DayOfWeek"] = d.Date.dt.dayofweek + 1
        d["Open"] = (d.DayOfWeek != 7).astype(int)
        week = (d.Date - dates[0]).dt.days // 7
        d["Promo"] = ((week % 2 == 0) & (d.DayOfWeek <= 5)).astype(int)
        d["StateHoliday"] = np.where(rng.random(len(d)) < 0.02, "a", "0")
        d["Open"] = np.where(d.StateHoliday != "0", 0, d.Open)
        d["SchoolHoliday"] = (d.Date.dt.month.isin([7, 8])).astype(int)
        season = 1 + 0.25 * (d.Date.dt.month == 12) + 0.1 * np.sin(2 * np.pi * d.Date.dt.dayofyear / 365)
        dow = d.DayOfWeek.map({1: 1.2, 2: 1.0, 3: 0.95, 4: 0.95, 5: 1.05, 6: 0.9, 7: 0})
        mu = base * season * dow * (1 + 0.3 * d.Promo)
        d["Sales"] = np.where(d.Open == 1, np.round(mu * rng.lognormal(0, 0.1, len(d))), 0).astype(int)
        d["Customers"] = (d.Sales / 9).astype(int)
        rows.append(d)
    full = pd.concat(rows, ignore_index=True)
    train = full[full.Date <= "2015-07-31"].reset_index(drop=True)
    test = full[full.Date > "2015-07-31"].drop(columns=["Sales", "Customers"]).reset_index(drop=True)
    test.insert(0, "Id", np.arange(1, len(test) + 1))
    return train, test, store
