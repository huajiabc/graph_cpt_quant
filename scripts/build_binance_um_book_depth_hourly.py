from pressure_graph.binance_um_book_depth_hourly import (
    BinanceBookDepthHourlyConfig,
    build_hourly_book_depth_panel,
)
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)


if __name__ == "__main__":
    manifest = build_hourly_book_depth_panel(
        list(FROZEN_SYMBOLS), BinanceBookDepthHourlyConfig(workers=8)
    )
    print(manifest.to_string(index=False))
