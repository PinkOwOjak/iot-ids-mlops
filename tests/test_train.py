import json
from pathlib import Path


def test_training_passes_quality_gate(trained):
    cfg, result = trained
    assert result["passed"], result["metrics"]
    assert result["metrics"]["mcc"] >= cfg.min_mcc


def test_artifacts_are_written(trained):
    cfg, _ = trained
    out = Path(cfg.output_dir)
    for name in ["model.joblib", "metadata.json", "reference_profile.json"]:
        assert (out / name).exists(), name
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["features"]
    assert meta["mlflow_run_id"]
    assert "ip.src_host" not in meta["features"]
