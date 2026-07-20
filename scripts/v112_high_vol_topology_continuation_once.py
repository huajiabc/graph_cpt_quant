from pressure_graph.reports.v112_high_vol_topology_continuation import (
    write_v112_high_vol_topology_continuation,
)


if __name__ == "__main__":
    for name, path in write_v112_high_vol_topology_continuation().items():
        print(f"{name}={path}")
