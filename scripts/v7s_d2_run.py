"""Driver for v7S Direction D2 — CVD-confirmed relative-value pair short.

Reads the v0.3 features parquet + the continuous Binance CVD shards
written by ``scripts/v7s_continuous_cvd_backfill.py``. Bars without
matching CVD data fail the CVD gate closed, so D2 is strictly dependent
on the P1 backfill catching up.

CLI: ``python scripts/v7s_d2_run.py --config configs/v0_3.yaml``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.config import load_config
from pressure_graph.io import read_parquet
from pressure_graph.reports.v7s_direction_d2_cvd_pair import D2Config, write_direction_d2


def main() -> None:
    parser = argparse.ArgumentParser(description="v7S Direction D2 — CVD-confirmed pair short.")
    parser.add_argument("--config", type=Path, default=Path("configs/v0_3.yaml"))
    parser.add_argument("--report-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    feature_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments_candidate = config.paths.data_root / "raw" / "bybit" / "instruments.parquet"
    instruments = read_parquet(instruments_candidate) if instruments_candidate.exists() else pd.DataFrame()

    cfg = D2Config(report_root=args.report_root) if args.report_root else D2Config()

    print(f"v7S D2 features: {feature_path} (exists={feature_path.exists()})", flush=True)
    print(f"v7S D2 continuous_root: {cfg.continuous_root} (exists={cfg.continuous_root.exists()})", flush=True)
    outputs = write_direction_d2(feature_path, instruments, config, cfg)
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
