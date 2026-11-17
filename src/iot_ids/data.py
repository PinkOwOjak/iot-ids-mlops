"""Data loading, feature engineering, cleaning and splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig

# Linux assigns client-side (ephemeral) ports from 32768 upwards. A destination
# port in that range just identifies one connection, so memorising its exact
# value is leakage; a port below it identifies the targeted service (80 = HTTP,
# 1883 = MQTT, 502 = Modbus), which is genuine behavioural signal.
EPHEMERAL_PORT_MIN = 32768
EPHEMERAL_BUCKET = -1


def _service_port(port: pd.Series) -> pd.Series:
    port = pd.to_numeric(port, errors="coerce").fillna(0)
    return port.where(port < EPHEMERAL_PORT_MIN, EPHEMERAL_BUCKET)


# Derived feature name -> (raw source column, transform).
DERIVED_FEATURES = {
    "tcp.dstport_service": ("tcp.dstport", _service_port),
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Used identically at training and serving time."""
    df = df.copy()
    for name, (source, fn) in DERIVED_FEATURES.items():
        if source in df.columns:
            df[name] = fn(df[source])
    return df


def raw_inputs(features: list[str]) -> list[str]:
    """Raw columns a caller must send to compute the given model features."""
    return [DERIVED_FEATURES[f][0] if f in DERIVED_FEATURES else f for f in features]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def select_features(df: pd.DataFrame, cfg: DataConfig) -> list[str]:
    """Return the feature columns: the configured list, or every numeric
    column that is neither the target nor a known leaky identifier."""
    if cfg.features:
        missing = [c for c in cfg.features if c not in df.columns]
        if missing:
            raise KeyError(f"Configured features not in data: {missing}")
        return list(cfg.features)
    excluded = set(cfg.drop_columns) | {cfg.target}
    candidates = [c for c in df.columns if c not in excluded]
    numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
    # Keep columns that are mostly numeric and not constant.
    keep = [c for c in candidates if numeric[c].notna().mean() > 0.95 and numeric[c].nunique() > 1]
    return keep


def to_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")


def prepare(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Clean the frame and return (X, y, feature_names)."""
    if cfg.target not in df.columns:
        raise KeyError(f"Target column '{cfg.target}' not found")
    df = add_derived(df.drop_duplicates())
    features = select_features(df, cfg)
    return to_matrix(df, features), df[cfg.target].astype(int), features


def split(X: pd.DataFrame, y: pd.Series, cfg: DataConfig):
    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
