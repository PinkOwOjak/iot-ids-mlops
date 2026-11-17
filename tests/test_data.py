import pandas as pd
import pytest

from iot_ids.config import DataConfig
from iot_ids.data import prepare


def test_leaky_identifiers_are_dropped(sample_csv):
    leaky = ["frame.time", "ip.src_host", "ip.dst_host", "tcp.seq", "tcp.checksum", "Attack_type"]
    cfg = DataConfig(path=str(sample_csv), drop_columns=leaky)
    X, y, features = prepare(pd.read_csv(sample_csv), cfg)
    for leaky in ["ip.src_host", "tcp.seq", "tcp.checksum", "Attack_type", "Attack_label"]:
        assert leaky not in features
    assert "tcp.len" in features
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0


def test_explicit_feature_list_is_respected(sample_csv):
    cfg = DataConfig(path=str(sample_csv), features=["tcp.len", "tcp.flags", "mqtt.len"])
    _, _, features = prepare(pd.read_csv(sample_csv), cfg)
    assert features == ["tcp.len", "tcp.flags", "mqtt.len"]


def test_missing_feature_raises(sample_csv):
    cfg = DataConfig(path=str(sample_csv), features=["does.not.exist"])
    with pytest.raises(KeyError):
        prepare(pd.read_csv(sample_csv), cfg)


def test_ephemeral_ports_are_bucketed():
    from iot_ids.data import EPHEMERAL_BUCKET, add_derived, raw_inputs

    df = pd.DataFrame({"tcp.dstport": [80, 1883, 47924, 60944, None]})
    out = add_derived(df)["tcp.dstport_service"].tolist()
    assert out == [80, 1883, EPHEMERAL_BUCKET, EPHEMERAL_BUCKET, 0]
    assert raw_inputs(["tcp.len", "tcp.dstport_service"]) == ["tcp.len", "tcp.dstport"]


def test_raw_destination_port_is_never_a_feature(sample_csv):
    cfg = DataConfig(path=str(sample_csv), drop_columns=["tcp.dstport", "Attack_type", "ip.src_host", "frame.time"])
    _, _, features = prepare(pd.read_csv(sample_csv), cfg)
    assert "tcp.dstport" not in features
    assert "tcp.dstport_service" in features
