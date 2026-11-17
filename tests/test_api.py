from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iot_ids import api


@pytest.fixture(scope="module")
def client(trained):
    cfg, _ = trained
    api.load_model(Path(cfg.output_dir))
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def features(client):
    # Raw inputs the caller must send (derived features are computed server-side).
    return client.get("/model-info").json()["required_inputs"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client, features):
    info = client.get("/model-info").json()
    assert "tcp.dstport_service" in info["features"]
    assert "tcp.dstport" in features  # raw port is what callers send
    assert "mcc" in info["metrics"]


def test_predict_attack_and_normal(client, features):
    attack = {f: 0.0 for f in features} | {"tcp.len": 3, "tcp.flags": 2, "tcp.connection.syn": 1, "mqtt.len": 0}
    normal = {f: 0.0 for f in features} | {
        "tcp.len": 60, "tcp.flags": 24, "tcp.connection.syn": 0, "mqtt.len": 20, "tcp.dstport": 443,
    }
    attack = {k: v for k, v in attack.items() if k in features}
    normal = {k: v for k, v in normal.items() if k in features}
    resp = client.post("/predict", json={"records": [attack, normal]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["label"] for p in preds] == [1, 0]
    assert 0.0 <= preds[0]["attack_probability"] <= 1.0


def test_predict_rejects_missing_features(client):
    resp = client.post("/predict", json={"records": [{"tcp.len": 5}]})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    text = client.get("/metrics/").text
    assert "iot_ids_predictions_total" in text
