from pressure_graph.reports.v207_unhedged_flow_exhaustion import (
    write_v207_diagnostic,
)


if __name__ == "__main__":
    for name, path in write_v207_diagnostic().items():
        print(f"{name}: {path}")
