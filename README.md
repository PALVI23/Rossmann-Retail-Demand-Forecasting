# End-to-end retail demand forecasting: from backtest to API

![ci](https://github.com/YOUR-USERNAME/forecast/actions/workflows/ci.yml/badge.svg)

A production-style forecasting system for **Rossmann** (1,115 drug stores, about 1M daily records). It forecasts daily sales 48 days ahead
for every store. It covers the full lifecycle, not just a notebook model:

1. **Data and EDA:** weekday, promotion and December effects, a 30× spread in store size, and a known data gap for 180 refurbished stores.
2. **Rolling-origin backtesting:** 3 folds, each with a 48-day horizon (the Kaggle test horizon). Store-level statistics are fit only on data before each cutoff (enforced by a unit test), and the boosting rounds are tuned on an inner window that ends before every evaluation fold.
3. **Models:** a seasonal-naive baseline, a store × weekday × promo median, and a **global LightGBM** on log sales with calendar, promo-cycle, competition, Promo2 and neighbouring-day holiday features. Includes a feature-group ablation and a bias correction fit on earlier folds.
4. **Uncertainty:** **split-conformal prediction intervals** (80% / 95%), with coverage checked on a later, unseen fold, overall and per store type.
5. **Official score:** a late submission to the Kaggle leaderboard, scored on the hidden test set.
6. **Serving:** a **FastAPI** service (`/predict`, `/health`, `/model-info`) that returns forecasts with intervals. Training and serving share one feature pipeline, so there's no train/serve skew. Includes a **Dockerfile**.
7. **Quality:** pytest unit and API tests run by **GitHub Actions CI** on every push, on synthetic data, so CI needs no download.
8. **Monitoring:** feature drift via the population stability index (PSI), comparing the training period with the forecast period.

## Results
> Filled in from `reports/results_summary.md` after the Colab run.

## Run it
**Colab (CPU):** upload the `forecast` folder to Google Drive, accept the [competition rules](https://www.kaggle.com/competitions/rossmann-store-sales/rules),
make sure the Colab secret `KAGGLE_API_TOKEN` exists, then run notebooks 01 → 02 → 03 (about 3 + 25 + 10 min).

**API locally:**
```bash
pip install -r requirements.txt
MODEL_DIR=artifacts/model uvicorn app.main:app --port 8000      # then open http://localhost:8000/docs
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"rows":[{"store":1,"date":"2015-08-03","promo":1},{"store":1,"date":"2015-08-04","promo":1}]}'
```
**Docker:** `docker build -t rossmann-forecast . && docker run -p 8000:8000 rossmann-forecast`

**Tests:** `pytest -q tests`. Smoke-run every notebook on synthetic data: `python scripts/smoke_test.py`.

## Layout
```
src/forecast/  data.py  features.py (shared by training + API)  models.py (baselines, LightGBM, backtest, conformal, PSI)
app/main.py    FastAPI service          Dockerfile, requirements-serve.txt
tests/         unit + API tests         .github/workflows/ci.yml
notebooks/     01_data_eda  02_backtest_intervals  03_final_submit_serve_monitor
```
