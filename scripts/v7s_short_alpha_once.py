"""One-shot driver for the v7S Short Alpha Exploration lane.

Reads the v0.3 features parquet + capacity trade cache + v11 cic_event
orderflow parquet, runs Direction E (strict CIC-failure-confirmed short)
across the top-30 universe, and writes the docx-mandated ten outputs to
``reports/v7s_short_alpha/E_cic_failure_confirmed/``.

This script intentionally avoids the project CLI so an A100 box without
the latest pipeline.py / cli.py edits can still execute the lane —
useful when graph_cpt_quant is deployed as a tarball rather than a git
clone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.config import load_config
from pressure_graph.io import read_parquet
from pressure_graph.reports.v7s_short_alpha import (
    V7SConfig,
    write_v7s_short_alpha,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="v7S Short Alpha Exploration — Direction E one-shot.")
    parser.add_argument("--config", type=Path, default=Path("configs/v0_3.yaml"), help="v0.3 base config.")
    parser.add_argument("--features", type=Path, default=None, help="Override features parquet path.")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/v7s_short_alpha"),
        help="Output root (per-direction subdir will be created under this).",
    )
    parser.add_argument(
        "--orderflow-event-path",
        type=Path,
        default=Path("data/orderflow_history/binance_um/cic_event_orderflow.parquet"),
        help="v11 cic_event_orderflow parquet (sell-flow gate source).",
    )
    parser.add_argument(
        "--sell-flow-fail-open",
        action="store_true",
        help="Pass sell-flow gate even when orderflow is missing (audit_reason will flag it).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    feature_path = args.features or (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )

    instruments_candidate = config.paths.data_root / "raw" / "bybit" / "instruments.parquet"
    instruments = read_parquet(instruments_candidate) if instruments_candidate.exists() else pd.DataFrame()

    cfg = V7SConfig(
        report_root=args.report_root,
        orderflow_event_path=args.orderflow_event_path,
        e_sell_flow_fail_open=args.sell_flow_fail_open,
    )

    print(f"v7S features: {feature_path} (exists={feature_path.exists()})", flush=True)
    print(f"v7S trade cache: {cfg.trade_cache_path} (exists={cfg.trade_cache_path.exists()})", flush=True)
    print(f"v7S orderflow event: {cfg.orderflow_event_path} (exists={cfg.orderflow_event_path.exists()})", flush=True)
    print(f"v7S sell_flow_fail_open: {cfg.e_sell_flow_fail_open}", flush=True)

    outputs = write_v7s_short_alpha(feature_path, instruments, config, cfg)
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
