from pressure_graph.reports.v2336_liquidation_price_mechanism_pilot import (
    write_v2336,
)


if __name__ == "__main__":
    for name, path in write_v2336().items():
        print(f"{name}: {path}")
