import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from forecast import models as M
    from forecast.data import make_synthetic

    train, _, store = make_synthetic(8, seed=2)
    m = M.fit_lgbm(train, store, num_boost_round=100)
    d = tmp_path_factory.mktemp("model")
    M.save_model(m, store, {0.8: 0.1, 0.95: 0.2}, {"version": "test", "trained_until": "2015-07-31"}, d)
    import os
    os.environ["MODEL_DIR"] = str(d)
    import app.main as api
    importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["n_stores"] == 8


def test_predict_batch_with_intervals(client):
    rows = [{"store": 1, "date": f"2015-08-0{d}", "promo": int(d < 5)} for d in range(1, 8)]
    rows[6]["open"] = 0
    r = client.post("/predict", json={"rows": rows})
    assert r.status_code == 200
    p = r.json()["predictions"]
    assert len(p) == 7 and p[6]["sales"] == 0
    assert all(x["lower_80"] <= x["sales"] <= x["upper_80"] <= x["upper_95"] for x in p[:6])


def test_validation(client):
    assert client.post("/predict", json={"rows": [{"store": 999, "date": "2015-08-01"}]}).status_code == 422
    assert client.post("/predict", json={"rows": [{"store": 1, "date": "2015-08-01", "state_holiday": "x"}]}).status_code == 422
    assert client.post("/predict", json={"rows": []}).status_code == 422
