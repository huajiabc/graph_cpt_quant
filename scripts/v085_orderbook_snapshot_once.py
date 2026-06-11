from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.config import load_config  # noqa: E402
from pressure_graph.orderbook import OrderbookRunConfig, write_v085_orderbook_snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one v0.8.5 orderbook snapshot shadow refresh."
    )
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--report-root", default="reports/v0_8_5_orderbook_snapshot")
    parser.add_argument("--orderbook-root", default="data/orderbook/v0_8_5/bybit")
    parser.add_argument("--demand-queue-path", default="data/orderflow/demand_queue.parquet")
    parser.add_argument(
        "--live-feature-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--max-symbols", type=int, default=10)
    parser.add_argument("--depth-limit", type=int, default=200)
    parser.add_argument("--retain-days", type=int, default=7)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    run_config = OrderbookRunConfig(
        report_root=Path(args.report_root),
        orderbook_root=Path(args.orderbook_root),
        demand_queue_path=Path(args.demand_queue_path),
        live_feature_path=Path(args.live_feature_path),
        top_n=args.top_n,
        max_symbols=None if args.max_symbols == 0 else args.max_symbols,
        depth_limit=args.depth_limit,
        retain_days=args.retain_days,
    )
    outputs = write_v085_orderbook_snapshot(base_config, run_config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
