from pressure_graph.reports.v2341_liquidation_graph_bucket_independent_audit import (
    write_v2341,
)


if __name__ == "__main__":
    for name, path in write_v2341().items():
        print(f"{name}: {path}")
