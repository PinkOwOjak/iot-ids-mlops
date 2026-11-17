import yaml

from iot_ids.config import load_config
from iot_ids.select_features import select, write_features_to_config


def test_selects_small_subset_close_to_full_mcc(config_path):
    cfg = load_config(config_path)
    result = select(cfg, tolerance=0.01)
    assert 1 <= len(result["selected"]) <= len(result["shap_ranking"])
    assert result["selected_mcc"] >= result["all_features_mcc"] - 0.01
    assert "ip.src_host" not in result["shap_ranking"]


def test_write_features_keeps_comments(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("data:\n  # keep me\n  path: x.csv\n  features: []\n")
    write_features_to_config(cfg, ["tcp.len", "mqtt.len"])
    text = cfg.read_text()
    assert "# keep me" in text
    assert yaml.safe_load(text)["data"]["features"] == ["tcp.len", "mqtt.len"]
