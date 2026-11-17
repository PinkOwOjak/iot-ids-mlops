"""Find the smallest feature set that keeps full-model performance, using SHAP.

Method (all on the TRAINING split only, so the test set never influences which
features are chosen):
  1. Split the training data again into fit / validation parts.
  2. Fit a random forest on all leakage-safe features and rank features by
     mean |SHAP value| on the fit part.
  3. Add features in ranked order, fitting the configured model on the top-k
     each time and scoring MCC on the validation part.
  4. Pick the smallest k whose MCC is within `--tolerance` of the all-features MCC.

Usage:
    python -m iot_ids.select_features --config config.yaml [--write-config]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from .config import Config, load_config
from .data import load_raw, prepare, split
from .train import build_model


def shap_ranking(X: pd.DataFrame, y: pd.Series, seed: int, sample: int = 2000) -> pd.Series:
    import shap

    forest = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)
    forest.fit(X, y)
    rows = X.sample(min(sample, len(X)), random_state=seed)
    values = shap.TreeExplainer(forest).shap_values(rows)
    if isinstance(values, list):  # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # newer SHAP: (rows, features, classes)
        values = values[:, :, 1]
    importance = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    return importance.sort_values(ascending=False)


def select(cfg: Config, data_path: str | None = None, tolerance: float = 0.005, max_rows: int = 200_000) -> dict:
    X, y, _ = prepare(load_raw(data_path or cfg.data.path), cfg.data)
    X_train, _, y_train, _ = split(X, y, cfg.data)  # same split as training; test part discarded
    if len(X_train) > max_rows:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_rows, stratify=y_train, random_state=cfg.data.random_state
        )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=cfg.data.random_state
    )

    ranking = shap_ranking(X_fit, y_fit, cfg.data.random_state)
    ranked = list(ranking.index)

    def score(cols: list[str]) -> float:
        model = build_model(cfg)
        model.fit(X_fit[cols], y_fit)
        return float(matthews_corrcoef(y_val, model.predict(X_val[cols])))

    full_mcc = score(ranked)
    curve = []
    chosen = ranked
    for k in range(1, len(ranked) + 1):
        mcc = score(ranked[:k])
        curve.append({"k": k, "mcc": round(mcc, 4)})
        if mcc >= full_mcc - tolerance:
            chosen = ranked[:k]
            break

    return {
        "all_features_mcc": round(full_mcc, 4),
        "tolerance": tolerance,
        "selected": chosen,
        "selected_mcc": curve[-1]["mcc"],
        "curve": curve,
        "shap_ranking": {f: round(float(v), 6) for f, v in ranking.items()},
    }


def write_features_to_config(config_path: str | Path, features: list[str]) -> None:
    """Replace the `features:` line in config.yaml, keeping comments intact."""
    path = Path(config_path)
    text = path.read_text()
    inline = "[" + ", ".join(json.dumps(f) for f in features) + "]"
    new_text, n = re.subn(r"(?m)^(\s*)features:.*$", lambda m: f"{m.group(1)}features: {inline}", text, count=1)
    if n != 1:
        raise ValueError("No 'features:' line found in config")
    path.write_text(new_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHAP-guided minimal feature selection.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="Override the CSV path from the config")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Allowed MCC drop vs all features")
    parser.add_argument("--write-config", action="store_true", help="Save the selected features into the config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg.data.features = []  # always rank from the full leakage-safe set
    result = select(cfg, args.data, args.tolerance)

    print("Top features by mean |SHAP|:")
    for name, value in list(result["shap_ranking"].items())[:10]:
        print(f"  {name:35s} {value:.4f}")
    print(f"\nAll {len(result['shap_ranking'])} features: validation MCC {result['all_features_mcc']}")
    for point in result["curve"]:
        print(f"  top-{point['k']:<3d} MCC {point['mcc']}")
    print(f"\nSelected {len(result['selected'])} features (MCC {result['selected_mcc']}): {result['selected']}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_selection.json").write_text(json.dumps(result, indent=2))

    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(f"{cfg.experiment}-feature-selection")
        with mlflow.start_run():
            mlflow.log_params({"tolerance": args.tolerance, "n_selected": len(result["selected"])})
            mlflow.log_metrics({"all_features_mcc": result["all_features_mcc"], "selected_mcc": result["selected_mcc"]})
            mlflow.log_dict(result, "feature_selection.json")
    except Exception as exc:  # tracking is optional for this step
        print(f"(MLflow logging skipped: {exc})", file=sys.stderr)

    if args.write_config:
        write_features_to_config(args.config, result["selected"])
        print(f"Wrote selected features to {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
