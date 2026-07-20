from pressure_graph.reports.v110_balanced_topology_break import (
    write_v110_balanced_topology_break,
)


if __name__ == "__main__":
    for name, path in write_v110_balanced_topology_break().items():
        print(f"{name}={path}")
