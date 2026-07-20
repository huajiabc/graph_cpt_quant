from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    write_v155_binance_one_percent_depth_imbalance,
)


if __name__ == "__main__":
    for name, path in write_v155_binance_one_percent_depth_imbalance().items():
        print(f"{name}: {path}")
