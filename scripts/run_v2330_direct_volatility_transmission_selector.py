from pressure_graph.reports.v2330_direct_volatility_transmission_selector import (
    write_v2330_direct_volatility_transmission_selector,
)


if __name__ == "__main__":
    for name, path in write_v2330_direct_volatility_transmission_selector().items():
        print(f"{name}: {path}")
