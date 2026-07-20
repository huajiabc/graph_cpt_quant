from pressure_graph.reports.v2340_liquidation_graph_volatility_transmission_pilot import (
    write_v2340,
)


if __name__ == "__main__":
    for name, path in write_v2340().items():
        print(f"{name}: {path}")
