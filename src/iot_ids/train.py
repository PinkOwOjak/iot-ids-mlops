"""Train, evaluate, track with MLflow, and export the model for serving.

Usage:
    python -m iot_ids.train --config config.yaml [--data path/to.csv]

Tracking server: set MLFLOW_TRACKING_URI (defaults to a local SQLite store).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.tree import DecisionTreeClassifier

from . import __version__
from .config import Config, load_config
from .data import load_raw, prepare, split
from .drift import build_reference

TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

MODELS = {"decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier}


def build_model(cfg: Config):
    params = dict(cfg.model.params)
    params.setdefault("random_state", cfg.data.random_state)
    if cfg.model.type == "random_forest":
        params.setdefault("n_jobs", -1)
    return MODELS[cfg.model.type](**params)


def evaluate(model, X_test, y_test) -> dict[str, float]:
    start = time.perf_counter()
    pred = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, pred),
        "latency_ms_per_1k_rows": 1000 * elapsed / max(len(X_test), 1) * 1000,
    }


def run(cfg: Config, data_path: str | None = None, config_path: str | None = None, run_name: str | None = None) -> dict:
    df = load_raw(data_path or cfg.data.path)
    X, y, features = prepare(df, cfg.data)
    X_train, X_test, y_train, y_test = split(X, y, cfg.data)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({"model_type": cfg.model.type, **cfg.model.params})
        mlflow.log_params({"n_features": len(features), "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.set_tag("features", ",".join(features))
        if config_path:
            mlflow.log_artifact(config_path)

        model = build_model(cfg)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        model_path = out / "model.joblib"
        joblib.dump(model, model_path)
        metrics["model_size_kb"] = model_path.stat().st_size / 1024
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        reference = build_reference(X_train)
        (out / "reference_profile.json").write_text(json.dumps(reference))
        mlflow.log_artifact(str(out / "reference_profile.json"))

        signature = infer_signature(X_test.head(50), model.predict(X_test.head(50)))
        log_kwargs = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            # Newer MLflow serialises with skops and needs the tree types allow-listed.
            log_kwargs["skops_trusted_types"] = TRUSTED_TYPES
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=cfg.registered_model_name,
            **log_kwargs,
        )

        metadata = {
            "model_type": cfg.model.type,
            "features": features,
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": active.info.run_id,
            "package_version": __version__,
            "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True), strict=True)},
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(out / "metadata.json"))

        passed = metrics["mcc"] >= cfg.min_mcc
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")

    return {"metrics": metrics, "passed": passed, "features": features}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the IoT intrusion detection model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="Override the CSV path from the config")
    parser.add_argument("--run-name", help="Name shown for this run in the MLflow UI")
    parser.add_argument(
        "--all-features", action="store_true",
        help="Ignore the configured feature list and use every leakage-safe column (used for synthetic CI data)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.all_features:
        cfg.data.features = []
    result = run(cfg, data_path=args.data, config_path=args.config, run_name=args.run_name)
    m = result["metrics"]
    print(f"Features ({len(result['features'])}): {', '.join(result['features'])}")
    print(
        f"MCC={m['mcc']:.4f}  F1={m['f1_macro']:.4f}  accuracy={m['accuracy']:.4f}  "
        f"size={m['model_size_kb']:.1f} KB  latency={m['latency_ms_per_1k_rows']:.2f} ms/1k rows"
    )
    if not result["passed"]:
        print(f"QUALITY GATE FAILED: MCC {m['mcc']:.4f} < {cfg.min_mcc}", file=sys.stderr)
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
