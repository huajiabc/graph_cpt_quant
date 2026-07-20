from pressure_graph.reports.v188_top_trader_absorption import (
    write_v188_top_trader_absorption,
)


if __name__ == "__main__":
    for name, path in write_v188_top_trader_absorption().items():
        print(f"{name}: {path}")
