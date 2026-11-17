import sys
from pathlib import Path

from iot_ids.data import prepare
from iot_ids.drift import build_reference, compare

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_sample_data import generate  # noqa: E402


def _features(df, cfg):
    X, _, _ = prepare(df, cfg.data)
    return X


def test_same_distribution_is_stable(trained):
    cfg, _ = trained
    ref = build_reference(_features(generate(3000, seed=1), cfg))
    scores = compare(ref, _features(generate(3000, seed=2), cfg))
    assert max(scores.values()) < 0.1


def test_shifted_distribution_is_flagged(trained):
    cfg, _ = trained
    ref = build_reference(_features(generate(3000, seed=1), cfg))
    scores = compare(ref, _features(generate(3000, seed=2, shift=True), cfg))
    assert max(scores.values()) > 0.2
