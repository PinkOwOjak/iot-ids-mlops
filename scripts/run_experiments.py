"""Reproduce the results table in the README on the real Edge-IIoTset.

Trains four models, all logged to MLflow under readable run names:
  1. leaky baseline     raw destination port kept (shows the leakage effect)
  2. dt-all-features    decision tree, every leakage-safe feature
  3. rf-all-features    random forest, every leakage-safe feature
  4. production         decision tree, SHAP-selected features from config.yaml
                        (the only run whose artifacts are written to models/)

Usage:
    python scripts/run_experiments.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import tempfile

from iot_ids.config import load_config
from iot_ids.train import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    base = load_config(args.config)
    tmp = tempfile.mkdtemp()

    leaky = copy.deepcopy(base)
    leaky.data.features = []
    leaky.data.drop_columns = [c for c in leaky.data.drop_columns if c != "tcp.dstport"]
    leaky.registered_model_name = None

    dt_all = copy.deepcopy(base)
    dt_all.data.features = []
    dt_all.registered_model_name = None

    rf_all = copy.deepcopy(dt_all)
    rf_all.model.type = "random_forest"
    rf_all.model.params = {"n_estimators": 200, "max_depth": 20}

    experiments = [
        ("leaky-baseline (raw port, NOT safe)", leaky, f"{tmp}/leaky"),
        ("dt-all-features", dt_all, f"{tmp}/dt"),
        ("rf-all-features", rf_all, f"{tmp}/rf"),
        ("production (SHAP-selected)", base, base.output_dir),
    ]

    rows = []
    for name, cfg, out in experiments:
        cfg.output_dir = out
        result = run(cfg, config_path=args.config, run_name=name)
        m = result["metrics"]
        rows.append(
            f"| {name} | {len(result['features'])} | {m['mcc']:.3f} | {m['f1_macro']:.3f} | {m['accuracy']:.3f} "
            f"| {m['model_size_kb']:.1f} KB | {m['latency_ms_per_1k_rows']:.2f} |"
        )

    print("| Run | Features | MCC | Macro F1 | Accuracy | Size | Latency (ms / 1k rows) |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
