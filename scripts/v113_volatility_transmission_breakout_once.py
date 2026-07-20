from pressure_graph.reports.v113_volatility_transmission_breakout import (
    write_v113_volatility_transmission_breakout,
)


if __name__ == "__main__":
    for name, path in write_v113_volatility_transmission_breakout().items():
        print(f"{name}={path}")
