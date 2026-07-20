from pressure_graph.reports.v111_sparse_topology_continuation import (
    write_v111_sparse_topology_continuation,
)


if __name__ == "__main__":
    for name, path in write_v111_sparse_topology_continuation().items():
        print(f"{name}={path}")
