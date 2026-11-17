"""Generate a small SYNTHETIC dataset with Edge-IIoTset-style columns.

Used only by tests and CI so the pipeline can run without the real dataset
(which is ~1 GB and must be downloaded separately). Numbers from this data mean
nothing; always report results from the real Edge-IIoTset.

Usage:
    python scripts/make_sample_data.py --rows 5000 --out data/sample.csv [--shift]

--shift produces "production" data with shifted traffic patterns, to demo the
drift check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int = 5000, seed: int = 0, shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attack = rng.random(rows) < 0.35
    s = 3.0 if shift else 1.0

    df = pd.DataFrame(
        {
            # Identifier-style columns (should be dropped by the pipeline).
            "frame.time": pd.date_range("2026-01-01", periods=rows, freq="s").astype(str),
            "ip.src_host": np.where(attack, "192.168.0.170", "192.168.0." + pd.Series(rng.integers(1, 100, rows)).astype(str)),
            "ip.dst_host": "192.168.0.128",
            "tcp.seq": rng.integers(0, 10**6, rows),
            "tcp.checksum": rng.integers(0, 65535, rows),
            # Behavioural features.
            "tcp.len": np.where(attack, rng.poisson(5, rows), rng.poisson(60 * s, rows)),
            "tcp.flags": np.where(attack, rng.choice([2, 4], rows), rng.choice([16, 24], rows)),
            "tcp.connection.syn": np.where(attack, rng.random(rows) < 0.8, rng.random(rows) < 0.05).astype(int),
            "tcp.connection.rst": np.where(attack, rng.random(rows) < 0.4, rng.random(rows) < 0.02).astype(int),
            "tcp.dstport": np.where(attack, rng.integers(1, 1024, rows), rng.choice([80, 443, 1883, 502], rows)),
            "udp.time_delta": rng.exponential(0.01 * s, rows),
            "dns.qry.name.len": rng.integers(0, 40, rows),
            "mqtt.len": np.where(attack, 0, rng.poisson(20 * s, rows)),
            "Attack_label": attack.astype(int),
            "Attack_type": np.where(attack, rng.choice(["DDoS_TCP", "Port_Scanning", "DDoS_HTTP"], rows), "Normal"),
        }
    )
    # A little label noise so the task is not trivially perfect.
    flip = rng.random(rows) < 0.01
    df.loc[flip, "Attack_label"] = 1 - df.loc[flip, "Attack_label"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--out", default="data/sample.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed, args.shift).to_csv(out, index=False)
    print(f"Wrote {args.rows} synthetic rows to {out}")


if __name__ == "__main__":
    main()
