from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.config import load_config  # noqa: E402
from pressure_graph.orderflow import OrderflowRunConfig, write_v08_orderflow_shadow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one v0.8 orderflow/taker-flow shadow refresh."
    )
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--report-root", default="reports/v0_8_orderflow_shadow")
    parser.add_argument("--orderflow-root", default="data/orderflow/v0_8/bybit")
    parser.add_argument("--demand-queue-path", default="data/orderflow/demand_queue.parquet")
    parser.add_argument("--source-report-root", default="reports/v0_7d2_cic_mir1_paper_live")
    parser.add_argument(
        "--live-feature-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--recent-trade-limit", type=int, default=1000)
    parser.add_argument("--retain-days", type=int, default=7)
    parser.add_argument("--report-lookback-days", type=int, default=7)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    run_config = OrderflowRunConfig(
        report_root=Path(args.report_root),
        orderflow_root=Path(args.orderflow_root),
        demand_queue_path=Path(args.demand_queue_path),
        source_report_root=Path(args.source_report_root),
        live_feature_path=Path(args.live_feature_path),
        top_n=args.top_n,
        max_symbols=None if args.max_symbols == 0 else args.max_symbols,
        recent_trade_limit=args.recent_trade_limit,
        retain_days=args.retain_days,
        report_lookback_days=args.report_lookback_days,
    )
    outputs = write_v08_orderflow_shadow(base_config, run_config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
