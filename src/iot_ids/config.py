"""Load and validate the YAML pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    path: str
    target: str = "Attack_label"
    drop_columns: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    type: str = "decision_tree"
    params: dict = field(default_factory=dict)


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    min_mcc: float = 0.8
    experiment: str = "iot-intrusion-detection"
    registered_model_name: str | None = None
    output_dir: str = "models"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    model = ModelConfig(**raw.get("model", {}))
    if model.type not in {"decision_tree", "random_forest"}:
        raise ValueError(f"Unsupported model type: {model.type}")
    mlflow_cfg = raw.get("mlflow", {})
    return Config(
        data=DataConfig(**raw["data"]),
        model=model,
        min_mcc=raw.get("quality_gate", {}).get("min_mcc", 0.8),
        experiment=mlflow_cfg.get("experiment", "iot-intrusion-detection"),
        registered_model_name=mlflow_cfg.get("registered_model_name"),
        output_dir=raw.get("output_dir", "models"),
    )
