"""FastAPI service that serves the trained model.

Run locally:
    uvicorn iot_ids.api:app --port 8000

Endpoints:
    GET  /health      liveness + whether a model is loaded
    GET  /model-info  features, metrics and MLflow run id of the loaded model
    POST /predict     classify one or more traffic records
    GET  /metrics     Prometheus metrics (request counts, latency, predictions)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .data import add_derived, raw_inputs, to_matrix

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))

PREDICTIONS = Counter("iot_ids_predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("iot_ids_predict_latency_seconds", "Latency of /predict")
ERRORS = Counter("iot_ids_request_errors_total", "Rejected prediction requests")

state: dict = {}


def load_model(model_dir: Path = MODEL_DIR) -> None:
    state["model"] = joblib.load(model_dir / "model.joblib")
    state["metadata"] = json.loads((model_dir / "metadata.json").read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    if (MODEL_DIR / "model.joblib").exists():
        load_model()
    yield
    state.clear()


app = FastAPI(title="IoT Intrusion Detection API", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class PredictRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    label: int
    attack_probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_run_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    if "metadata" not in state:
        raise HTTPException(503, "No model loaded")
    return state["metadata"] | {"required_inputs": raw_inputs(state["metadata"]["features"])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in state:
        raise HTTPException(503, "No model loaded")
    features = state["metadata"]["features"]
    required = raw_inputs(features)
    missing = sorted({f for r in req.records for f in required if f not in r})
    if missing:
        ERRORS.inc()
        raise HTTPException(422, f"Missing features: {missing}")

    with LATENCY.time():
        X = to_matrix(add_derived(pd.DataFrame(req.records)), features)
        model = state["model"]
        labels = model.predict(X)
        classes = list(model.classes_)
        proba = model.predict_proba(X)[:, classes.index(1)] if 1 in classes else [0.0] * len(X)

    for label in labels:
        PREDICTIONS.labels(label=str(int(label))).inc()

    return PredictResponse(
        predictions=[
            Prediction(label=int(lbl), attack_probability=round(float(p), 4))
            for lbl, p in zip(labels, proba, strict=True)
        ],
        model_run_id=state["metadata"]["mlflow_run_id"],
    )
