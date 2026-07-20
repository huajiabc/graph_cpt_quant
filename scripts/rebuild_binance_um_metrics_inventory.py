from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.binance_metrics_history import (
    BinanceMetricsConfig,
    write_binance_metrics_inventory,
)


MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
OUTPUT_ROOT = Path("data/external/binance_um_metrics_5m")


if __name__ == "__main__":
    membership = pd.read_csv(MEMBERSHIP_PATH, usecols=["symbol"])
    expected = sorted({*membership["symbol"].astype(str).unique(), "BTCUSDT"})
    manifest = write_binance_metrics_inventory(
        BinanceMetricsConfig(output_root=OUTPUT_ROOT),
        expected_symbols=expected,
    )
    print(manifest.to_string(index=False))
