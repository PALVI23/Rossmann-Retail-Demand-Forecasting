"""Forecast API.  Run:  MODEL_DIR=artifacts/model uvicorn app.main:app --port 8000
POST /predict with one or more (store, date) rows -> point forecast + 80% / 95% conformal intervals."""
from __future__ import annotations

import os
import sys
import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast.models import interval, load_model  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).resolve().parents[1] / "artifacts" / "model"))
app = FastAPI(title="Rossmann demand forecast", version="1.0")
_state: dict = {}


def state():
    if not _state:
        model, store, meta = load_model(MODEL_DIR)
        _state.update(model=model, store=store, meta=meta, stores=set(store.Store))
    return _state


class Row(BaseModel):
    store: int = Field(..., ge=1, examples=[1])
    date: dt.date = Field(..., examples=["2015-08-01"])
    open: int = Field(1, ge=0, le=1)
    promo: int = Field(0, ge=0, le=1)
    state_holiday: Literal["0", "a", "b", "c"] = "0"
    school_holiday: int = Field(0, ge=0, le=1)


class PredictRequest(BaseModel):
    rows: list[Row] = Field(..., min_length=1, max_length=10_000,
                            description="Send consecutive days per store when possible: neighbouring-day features "
                                        "(holiday/closed tomorrow, promo run) are computed within the request.")


@app.get("/health")
def health():
    s = state()
    return {"status": "ok", "model_version": s["meta"].get("version"), "n_stores": len(s["stores"])}


@app.get("/model-info")
def model_info():
    m = state()["meta"]
    return {k: m.get(k) for k in ["version", "trained_until", "backtest", "conformal_coverage", "kaggle_score"]}


@app.post("/predict")
def predict(req: PredictRequest):
    s = state()
    df = pd.DataFrame([r.model_dump() for r in req.rows]).rename(columns={
        "store": "Store", "date": "Date", "open": "Open", "promo": "Promo", "state_holiday": "StateHoliday",
        "school_holiday": "SchoolHoliday"})
    unknown = sorted(set(df.Store) - s["stores"])
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown store id(s): {unknown[:10]}")
    df["Date"] = pd.to_datetime(df.Date)
    yhat = s["model"].predict(df, s["store"])
    out = []
    q = s["meta"]["conformal_log_q"]
    bounds = {lv: interval(yhat, q[lv]) for lv in q}
    for i, r in df.iterrows():
        rec = {"store": int(r.Store), "date": r.Date.date().isoformat(), "sales": round(float(yhat[i]), 1)}
        for lv, (lo, hi) in bounds.items():
            pct = int(float(lv) * 100)
            rec[f"lower_{pct}"], rec[f"upper_{pct}"] = round(float(lo[i]), 1), round(float(hi[i]), 1)
        out.append(rec)
    return {"predictions": out}
