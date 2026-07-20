from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.binance_aggtrade_event_history import (
    AggTradeArchiveEventConfig,
    AggTradeEventConfig,
    build_extreme_overshoot_tasks,
    collect_aggtrade_event_archives,
    collect_aggtrade_event_windows,
)


FEATURE_EVENTS_PATH = Path(
    "reports/v20_0_reference_price_transmission_feature_audit/"
    "candidate_feature_events.parquet"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill exact aggTrade windows for frozen overshoot events."
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--request-interval", type=float, default=0.65)
    parser.add_argument(
        "--mode", choices=["archive", "rest"], default="archive"
    )
    args = parser.parse_args()
    events = pd.read_parquet(FEATURE_EVENTS_PATH)
    tasks = build_extreme_overshoot_tasks(events)
    if args.mode == "archive":
        features, manifest = collect_aggtrade_event_archives(
            tasks,
            AggTradeArchiveEventConfig(),
            max_archives=args.max_archives,
        )
    else:
        features, manifest = collect_aggtrade_event_windows(
            tasks,
            AggTradeEventConfig(request_interval_seconds=args.request_interval),
            max_tasks=args.max_tasks,
        )
    status_column = "status"
    print(
        f"tasks={len(tasks)} stored={len(features)} "
        f"completed_now={manifest[status_column].eq('complete').sum() if len(manifest) else 0} "
        f"errors_now={manifest[status_column].eq('error').sum() if len(manifest) else 0}"
    )


if __name__ == "__main__":
    main()
