from pressure_graph.reports.v2321_alt_first_volatility_ignition_breakout import (
    write_v2321_alt_first_volatility_ignition_breakout,
)


if __name__ == "__main__":
    for name, path in write_v2321_alt_first_volatility_ignition_breakout().items():
        print(f"{name}: {path}")
