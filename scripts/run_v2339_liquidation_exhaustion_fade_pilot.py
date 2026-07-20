from pressure_graph.reports.v2339_liquidation_exhaustion_fade_pilot import (
    write_v2339,
)


if __name__ == "__main__":
    for name, path in write_v2339().items():
        print(f"{name}: {path}")
