"""Generates the notebooks in ../notebooks.   Run: python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# ---------- Setup: run this cell first (CPU runtime is enough) ----------
import os, sys, subprocess, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
SMOKE = os.environ.get("FC_SMOKE") == "1"
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
if SMOKE:
    PROJECT_DIR = os.environ["FC_PROJECT_DIR"]
elif ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    import glob
    hits = sorted(glob.glob("/content/drive/MyDrive/**/forecast/src/forecast/models.py", recursive=True), key=len)
    assert hits, "Upload the 'forecast' folder to your Google Drive first."
    PROJECT_DIR = str(Path(hits[0]).parents[2])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabulate", "fastapi", "uvicorn", "httpx"], check=True)
else:
    PROJECT_DIR = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
sys.path.insert(0, f"{PROJECT_DIR}/src"); sys.path.insert(0, PROJECT_DIR)
ART, FIG = f"{PROJECT_DIR}/artifacts", f"{PROJECT_DIR}/reports/figures"
DATA = "/content/data/rossmann" if ON_COLAB else f"{PROJECT_DIR}/data"
for d in (ART, FIG, DATA): os.makedirs(d, exist_ok=True)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
from forecast import data as D, models as M
from forecast.features import FEATURES, build_features, fit_store_stats
if SMOKE:
    train, test, store = D.make_synthetic(25)
else:
    train, test, store = D.load(D.download(DATA))
def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=160, bbox_inches="tight"); plt.show()
def save_json(o, name):
    json.dump(o, open(f"{ART}/{name}.json", "w"), indent=2, default=str)
def load_json(name):
    return json.load(open(f"{ART}/{name}.json"))
print(f"train {train.shape} {train.Date.min().date()}..{train.Date.max().date()} | test {test.shape} | stores {store.shape[0]}")'''


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    for kind, src in cells:
        src = src.strip("\n")
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return book


NB1 = [
("md", """# 01 · Data and exploratory analysis
**Project:** an end-to-end retail demand forecasting system. Forecasting, uncertainty, an API, CI and monitoring.
**Data:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales): 1,115 stores and about 1M daily records (2013–2015).
The task is to forecast daily sales for the next **48 days**.

> Before running: accept the competition rules on Kaggle (Rules tab → *I Understand and Accept*). The Colab secret
> `KAGGLE_API_TOKEN` from Project 1 is reused. **Runtime: CPU** (about 3 min)."""),
("code", SETUP),
("md", "## 1. Structure and data quality"),
("code", '''print("days per store:", train.groupby("Store").size().describe()[["min", "50%", "max"]].to_dict())
closed = (train.Open == 0).mean(); zero_open = ((train.Open == 1) & (train.Sales == 0)).sum()
print(f"closed days: {closed:.1%} | open days with zero sales: {zero_open}")
n_days = train.groupby("Store").Date.nunique()
gap_stores = n_days[n_days < n_days.max()].index
print(f"{len(gap_stores)} stores have missing dates (Rossmann's known mid-2014 gap for refurbished stores)")
store.isna().mean().round(3).to_frame("missing share")'''),
("code", '''op = train[train.Open == 1]
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
op.groupby("Date").Sales.mean().rolling(7).mean().plot(ax=ax[0, 0]); ax[0, 0].set_title("Mean daily sales per open store (7-day avg)")
op.groupby("DayOfWeek").Sales.mean().plot.bar(ax=ax[0, 1], color="#4C72B0"); ax[0, 1].set_title("By day of week")
op.groupby(["Promo"]).Sales.mean().plot.bar(ax=ax[1, 0], color=["#BBBBBB", "#C44E52"]); ax[1, 0].set_title("Promo vs no promo")
op.merge(store[["Store", "StoreType"]]).groupby("StoreType").Sales.mean().plot.bar(ax=ax[1, 1], color="#55A868")
ax[1, 1].set_title("By store type"); savefig("01_patterns")'''),
("code", '''lift = op.groupby("Promo").Sales.mean(); print(f"promo days: +{lift[1]/lift[0]-1:.1%} sales on average")
per_store = op.groupby("Store").Sales.mean()
print(f"store mean sales: min {per_store.min():.0f}, median {per_store.median():.0f}, max {per_store.max():.0f}  -> "
      f"{per_store.max()/per_store.min():.0f}x spread: a global model needs store-level information")
dec = op[op.Date.dt.month == 12].groupby(op.Date.dt.day).Sales.mean()
dec.plot(figsize=(7, 3), marker="o", title="December: mean sales by day of month"); savefig("01_december")'''),
("code", '''save_json({"n_rows": len(train), "n_stores": int(store.shape[0]), "closed_share": float(closed),
           "promo_lift": float(lift[1] / lift[0] - 1), "store_spread": float(per_store.max() / per_store.min()),
           "n_gap_stores": int(len(gap_stores))}, "eda")
print("saved")'''),("md", """## Observations (fill in yourself)
* Which patterns (weekday, promo, seasonality, store type) must the model capture?
* Data-quality issues and how they're handled.""")]

NB2 = [
("md", """# 02 · Backtesting: baselines vs. a global LightGBM model, with prediction intervals
**Rolling-origin backtest:** 3 folds, each forecasting the 48 days after its cutoff (the same horizon as the Kaggle test).
Store statistics are fit only on data before each cutoff. The number of boosting rounds is chosen *once* on an inner
validation window that ends before every evaluation window, so no evaluation data is used for tuning. **Runtime: CPU**, about 20–30 min."""),
("code", SETUP),
("code", '''folds = M.make_folds(train, n_folds=3, horizon=48)
for f in folds: print(f["fold"], "train <=", f["cutoff"].date(), "| forecast", (f["cutoff"] + pd.Timedelta(days=1)).date(), "->", f["valid_end"].date())
# pick the number of boosting rounds on an inner window before the earliest evaluation fold
earliest = folds[-1]["cutoff"]
inner_hist = train[train.Date <= earliest - pd.Timedelta(days=48)]
inner_val = train[(train.Date > earliest - pd.Timedelta(days=48)) & (train.Date <= earliest)]
t0 = time.time()
m_inner = M.fit_lgbm(inner_hist, store, num_boost_round=300 if SMOKE else 4000, valid=inner_val)
N_ROUNDS = max(100, int(m_inner.booster.best_iteration * 1.1))
print(f"rounds = {N_ROUNDS}  ({time.time()-t0:.0f}s)")'''),
("md", "## 1. Backtest"),
("code", '''rows, fold_preds = [], {}
for f in folds:
    hist, fut = M.split(train, f)
    t0 = time.time()
    model = M.fit_lgbm(hist, store, num_boost_round=N_ROUNDS)
    preds = {"Seasonal naive (same weekday, last 6 wks)": M.baseline_last_weeks(hist, fut),
             "Store x weekday x promo median": M.baseline_store_dow_promo(hist, fut),
             "LightGBM (global)": model.predict(fut, store)}
    fold_preds[f["fold"]] = (fut, preds)
    for name, p in preds.items():
        rows.append({"fold": f["fold"], "model": name, **M.metrics(fut.Sales, p)})
    print(f"fold {f['fold']}: LightGBM RMSPE {M.rmspe(fut.Sales, preds['LightGBM (global)']):.4f}  ({time.time()-t0:.0f}s)")
bt = pd.DataFrame(rows)
summary = bt.groupby("model")[["rmspe", "wape", "mae"]].agg(["mean", "std"]).sort_values(("rmspe", "mean"))
bt.to_csv(f"{ART}/backtest_folds.csv", index=False); summary.round(4)'''),
("md", """## 2. Bias correction
RMSPE punishes over-forecasts more than under-forecasts, and back-transforming from log sales adds bias. A single multiplicative
factor is fit on the *earlier* folds and checked on the latest fold."""),
("code", '''def best_factor(y, p):
    grid = np.linspace(0.94, 1.04, 51)
    return float(grid[np.argmin([M.rmspe(y, p * g) for g in grid])])
fut1, pr1 = fold_preds[1]; fut2, pr2 = fold_preds[2]
FACTOR = best_factor(np.r_[fut1.Sales.values, fut2.Sales.values], np.r_[pr1["LightGBM (global)"], pr2["LightGBM (global)"]])
fut0, pr0 = fold_preds[0]
print(f"factor {FACTOR:.3f} (fit on folds 1-2) | fold 0 RMSPE: {M.rmspe(fut0.Sales, pr0['LightGBM (global)']):.4f} -> "
      f"{M.rmspe(fut0.Sales, pr0['LightGBM (global)'] * FACTOR):.4f}")'''),
("md", "## 3. Where are the errors? By forecast horizon, weekday and store type (latest fold)"),
("code", '''e = fut0[["Store", "Date", "DayOfWeek", "Sales", "Open"]].copy(); e["pred"] = pr0["LightGBM (global)"] * FACTOR
e = e[(e.Open == 1) & (e.Sales > 0)].merge(store[["Store", "StoreType"]])
e["ape"] = (e.Sales - e.pred).abs() / e.Sales; e["horizon"] = (e.Date - folds[0]["cutoff"]).dt.days
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
e.groupby("horizon").ape.mean().plot(ax=ax[0]); ax[0].set_title("Mean abs. % error vs days ahead"); ax[0].set_xlabel("days ahead")
e.groupby("DayOfWeek").ape.mean().plot.bar(ax=ax[1], color="#4C72B0"); ax[1].set_title("by weekday")
e.groupby("StoreType").ape.mean().plot.bar(ax=ax[2], color="#55A868"); ax[2].set_title("by store type")
savefig("02_error_breakdown")
worst = e.groupby("Store").ape.mean().sort_values(ascending=False).head(5); print("worst stores (MAPE):", worst.round(3).to_dict())'''),
("md", "## 4. Feature-group ablation (latest fold): what does each group contribute?"),
("code", '''groups = {
    "store history stats": [f for f in FEATURES if f.startswith("store_")],
    "neighbouring-day context": ["holiday_tomorrow", "holiday_yesterday", "closed_tomorrow", "closed_yesterday", "promo_day_index", "promo_tomorrow"],
    "Promo2 & competition": ["comp_months_open", "log_comp_distance", "CompetitionDistance", "Promo2", "promo2_weeks", "promo2_active_month"],
    "calendar": ["year", "month", "day", "weekofyear", "dayofyear", "days_to_month_end", "is_december", "days_to_christmas"]}
hist0, fut0_ = M.split(train, folds[0]); full_rmspe = M.rmspe(fut0_.Sales, pr0["LightGBM (global)"])
abl = [{"removed": "(none: full model)", "rmspe": full_rmspe, "delta": 0.0}]
for g, cols in groups.items():
    feats = [f for f in FEATURES if f not in cols]
    p = M.fit_lgbm(hist0, store, features=feats, num_boost_round=N_ROUNDS).predict(fut0_, store)
    abl.append({"removed": g, "rmspe": M.rmspe(fut0_.Sales, p), "delta": M.rmspe(fut0_.Sales, p) - full_rmspe}); print(abl[-1])
abl = pd.DataFrame(abl); abl.to_csv(f"{ART}/ablation.csv", index=False); abl.round(4)'''),
("md", """## 5. Prediction intervals: split conformal, with coverage checked on unseen data
Calibrate on fold 1 (residuals on the log scale), then measure how often fold 0's actual sales fall inside the intervals."""),
("code", '''Q = M.conformal_quantiles(fut1.Sales.values, pr1["LightGBM (global)"] * FACTOR, levels=(0.8, 0.95))
cov_rows = []
y0, p0 = fut0.Sales.values, pr0["LightGBM (global)"] * FACTOR
st0 = fut0[["Store"]].merge(store[["Store", "StoreType"]]).StoreType.values
for lv, q in Q.items():
    lo, hi = M.interval(p0, q)
    row = {"nominal": lv, "log_halfwidth": q, "coverage_overall": M.coverage(y0, lo, hi),
           "median_width_%": float(np.median(((hi - lo) / np.maximum(p0, 1))[y0 > 0]) * 100)}
    for t in sorted(set(st0)):
        row[f"coverage_type_{t}"] = M.coverage(y0, lo, hi, mask=(y0 > 0) & (st0 == t))
    cov_rows.append(row)
cov = pd.DataFrame(cov_rows); cov.round(3)'''),
("code", '''s = fut0.assign(pred=p0)[lambda d: d.Store == d.Store.iloc[0]].sort_values("Date")
lo80, hi80 = M.interval(s.pred.values, Q[0.8]); lo95, hi95 = M.interval(s.pred.values, Q[0.95])
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.fill_between(s.Date, lo95, hi95, alpha=.15, label="95% interval"); ax.fill_between(s.Date, lo80, hi80, alpha=.3, label="80% interval")
ax.plot(s.Date, s.pred, label="forecast"); ax.plot(s.Date, s.Sales, "k.", label="actual")
ax.set_title(f"Store {s.Store.iloc[0]}: 48-day forecast with conformal intervals (unseen fold)"); ax.legend(fontsize=8)
savefig("02_forecast_example")'''),
("code", '''imp = pd.Series(model.booster.feature_importance("gain"), index=model.features).sort_values().tail(15)
imp.plot.barh(figsize=(6, 5), color="#4C72B0", title="LightGBM feature importance (gain)"); savefig("02_importance")
save_json({"n_rounds": N_ROUNDS, "factor": FACTOR, "summary": {m: {k: float(v) for k, v in g.items()} for m, g in bt.groupby("model")[["rmspe", "wape"]].mean().iterrows()},
           "per_fold": bt.to_dict(orient="records"), "conformal_q": Q, "coverage": cov.to_dict(orient="records")}, "backtest")
print("saved")'''),
("md", """## Observations (fill in yourself)
* How much better than the baselines is the global model, and is it stable across folds?
* Which feature groups matter most? Do the intervals achieve their nominal coverage, and for which store types not?"""),
]

NB3 = [
("md", """# 03 · Final model, Kaggle submission, deployment and monitoring
Trains on all the data, scores the official test set on Kaggle (late submission), exports the model for the **FastAPI** service,
calls the running API, and runs a **drift check**. Writes `reports/results_summary.md`. **Runtime: CPU**, about 10 min."""),
("code", SETUP),
("code", '''bt = load_json("backtest")
t0 = time.time()
final = M.fit_lgbm(train, store, num_boost_round=bt["n_rounds"])
test_pred = final.predict(test, store) * bt["factor"]
sub = pd.DataFrame({"Id": test.Id, "Sales": test_pred}).sort_values("Id")
sub.to_csv(f"{ART}/submission.csv", index=False)
print(f"trained on {len(train):,} rows in {time.time()-t0:.0f}s; submission {sub.shape}"); sub.head()'''),
("md", "## 1. Official score: Kaggle late submission (private leaderboard = 52% of the test set)"),
("code", '''KAGGLE = {}
if not SMOKE:
    try:
        r = subprocess.run(["kaggle", "competitions", "submit", "-c", D.COMPETITION, "-f", f"{ART}/submission.csv",
                            "-m", f"LightGBM global model, {bt['n_rounds']} rounds, factor {bt['factor']:.3f}"],
                           capture_output=True, text=True); print(r.stdout[-300:], r.stderr[-300:])
        for _ in range(30):                      # wait for scoring
            time.sleep(10)
            out = subprocess.run(["kaggle", "competitions", "submissions", "-c", D.COMPETITION, "--csv"],
                                 capture_output=True, text=True).stdout
            import io
            subs = pd.read_csv(io.StringIO(out)); subs.columns = [c.strip() for c in subs.columns]
            latest = subs.iloc[0]
            if str(latest.get("status", "")).lower().endswith("complete") and pd.notna(latest.get("privateScore")):
                KAGGLE = {"public_rmspe": float(latest["publicScore"]), "private_rmspe": float(latest["privateScore"])}
                break
        print("Kaggle scores:", KAGGLE or "not available yet - check the competition's 'Submissions' page")
    except Exception as e:  # noqa: BLE001
        print("Submission failed:", e, "-> upload artifacts/submission.csv manually via 'Late Submission' on Kaggle")'''),
("md", "## 2. Export the model and serve it with FastAPI"),
("code", '''Q = {float(k): v for k, v in bt["conformal_q"].items()}
MODEL_DIR = f"{ART}/model"
M.save_model(final, store, Q, {"version": "1.0", "trained_until": str(train.Date.max().date()), "backtest": bt["summary"],
                                "conformal_coverage": bt["coverage"], "kaggle_score": KAGGLE, "factor": bt["factor"]}, MODEL_DIR)
os.environ["MODEL_DIR"] = MODEL_DIR
import importlib, app.main as api; importlib.reload(api)
from fastapi.testclient import TestClient          # in-process client: same code path as the deployed service
client = TestClient(api.app)
print(client.get("/health").json())
s1 = int(test.Store.iloc[0])
rows = [{"store": s1, "date": str(d.date()), "promo": int(p), "open": int(o)} for d, p, o in
        test[test.Store == s1].sort_values("Date")[["Date", "Promo", "Open"]].head(7).itertuples(index=False)]
pd.DataFrame(client.post("/predict", json={"rows": rows}).json()["predictions"])'''),
("md", """**Deploying it:** `docker build -t rossmann-forecast . && docker run -p 8000:8000 rossmann-forecast`, then open
`http://localhost:8000/docs` for the interactive Swagger UI. GitHub Actions runs the unit and API tests on every push."""),
("md", "## 3. Monitoring: has the input data drifted? (population stability index, training vs. test period)"),
("code", '''ref = build_features(train[train.Date > train.Date.max() - pd.Timedelta(days=365)], store, final.stats)
cur = build_features(test, store, final.stats)
num = [f for f in final.features if f not in ("StoreType", "Assortment", "StateHoliday", "Store")]
drift = pd.Series({f: M.psi(ref[f], cur[f]) for f in num}).sort_values(ascending=False)
drift.head(10).plot.barh(figsize=(6, 4), color=["#C44E52" if v > 0.25 else "#DD8452" if v > 0.1 else "#55A868" for v in drift.head(10)])
plt.axvline(0.1, ls="--", c="k", lw=1); plt.axvline(0.25, ls="--", c="r", lw=1); plt.gca().invert_yaxis()
plt.title("Feature drift (PSI): last year vs. forecast period"); savefig("03_drift")
drift.head(10).round(3).to_frame("PSI")'''),
("md", "## 4. Auto-generated results summary"),
("code", '''eda = load_json("eda"); abl = pd.read_csv(f"{ART}/ablation.csv"); folds = pd.read_csv(f"{ART}/backtest_folds.csv")
summ = folds.groupby("model")[["rmspe", "wape"]].agg(["mean", "std"]).round(4)
lg = summ.loc["LightGBM (global)"]; nb_ = summ.loc["Seasonal naive (same weekday, last 6 wks)"]
cov = pd.DataFrame(bt["coverage"])
lines = ["# Results summary (auto-generated by notebook 03)", "",
 f"* Data: {eda['n_rows']:,} store-days, {eda['n_stores']} stores; promo lift +{eda['promo_lift']:.1%}; {eda['store_spread']:.0f}x spread in store size",
 f"* **Backtest (3 rolling 48-day folds): LightGBM RMSPE {lg[('rmspe','mean')]:.4f} ± {lg[('rmspe','std')]:.4f} vs seasonal naive "
 f"{nb_[('rmspe','mean')]:.4f}** (WAPE {lg[('wape','mean')]:.4f} vs {nb_[('wape','mean')]:.4f})",
 f"* Bias-correction factor {bt['factor']:.3f} (fit on earlier folds)",
 "* Conformal coverage on unseen fold: " + ", ".join(f"{int(r['nominal']*100)}% nominal -> {r['coverage_overall']:.1%}" for r in bt["coverage"]),
 f"* **Kaggle (late submission): {KAGGLE}**" if KAGGLE else "* Kaggle score: (not available, see notebook)",
 f"* Drift: {int((drift > 0.25).sum())} features with PSI > 0.25, {int(((drift > 0.1) & (drift <= 0.25)).sum())} with 0.1-0.25",
 "", "## Backtest", summ.to_markdown(), "", "## Per fold", folds.round(4).to_markdown(index=False),
 "", "## Feature-group ablation (latest fold)", abl.round(4).to_markdown(index=False),
 "", "## Conformal intervals", cov.round(3).to_markdown(index=False),
 "", "## Top drift (PSI)", drift.head(10).round(3).to_frame("PSI").to_markdown()]
open(f"{PROJECT_DIR}/reports/results_summary.md", "w").write("\\n".join(lines)); print("\\n".join(lines[:9]))'''),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in [("01_data_eda", NB1), ("02_backtest_intervals", NB2), ("03_final_submit_serve_monitor", NB3)]:
        nbf.write(nb(cells), out / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main()
