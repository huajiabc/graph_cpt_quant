from __future__ import annotations

import argparse
from pathlib import Path

from pressure_graph.okx_liquidation_forward import (
    OkxLiquidationConfig,
    collect_okx_liquidation_snapshot,
)
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/okx_liquidation_forward"),
    )
    parser.add_argument("--symbols", nargs="*")
    args = parser.parse_args()
    symbols = args.symbols or ["BTCUSDT", *FROZEN_SYMBOLS]
    manifest = collect_okx_liquidation_snapshot(
        symbols,
        OkxLiquidationConfig(output_root=args.output_root),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
