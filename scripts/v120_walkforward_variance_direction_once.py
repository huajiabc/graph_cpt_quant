from pressure_graph.reports.v120_walkforward_variance_direction import (
    write_v120_walkforward_variance_direction,
)


if __name__ == "__main__":
    for name, path in write_v120_walkforward_variance_direction().items():
        print(f"{name}: {path}")
