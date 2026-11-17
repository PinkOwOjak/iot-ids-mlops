"""Data drift monitoring with the Population Stability Index (PSI).

At training time we save a reference profile: quantile bin edges and the share
of training rows in each bin, per feature. Later, new production data is binned
the same way and compared. Common rule of thumb:
    PSI < 0.1  -> stable
    0.1 - 0.2  -> moderate shift, watch
    PSI > 0.2  -> significant drift, investigate / retrain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import add_derived

EPS = 1e-6
DRIFT_THRESHOLD = 0.2


def build_reference(X: pd.DataFrame, bins: int = 10) -> dict:
    profile = {}
    for col in X.columns:
        values = X[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:  # constant column
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        profile[col] = {"edges": edges.tolist(), "expected": (counts / counts.sum()).tolist()}
    return profile


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compare(reference: dict, X_new: pd.DataFrame) -> dict[str, float]:
    scores = {}
    for col, ref in reference.items():
        if col not in X_new.columns:
            continue
        edges = np.asarray(ref["edges"])
        values = pd.to_numeric(X_new[col], errors="coerce").fillna(0.0).to_numpy()
        counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        actual = counts / max(counts.sum(), 1)
        scores[col] = psi(np.asarray(ref["expected"]), actual)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check new data for drift against the training reference.")
    parser.add_argument("csv", help="CSV of new (production) data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    args = parser.parse_args(argv)

    reference = json.loads((Path(args.model_dir) / "reference_profile.json").read_text())
    scores = compare(reference, add_derived(pd.read_csv(args.csv, low_memory=False)))
    drifted = {c: s for c, s in scores.items() if s > args.threshold}

    for col, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        flag = "DRIFT" if score > args.threshold else "ok"
        print(f"{col:35s} PSI={score:7.4f}  {flag}")
    print(f"\n{len(drifted)} of {len(scores)} features drifted (threshold {args.threshold}).")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
