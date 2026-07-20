from pressure_graph.reports.v109_graph_dispersion_spread import (
    write_v109_graph_dispersion_spread,
)


if __name__ == "__main__":
    for name, path in write_v109_graph_dispersion_spread().items():
        print(f"{name}={path}")
