from pressure_graph.reports.v2342_liquidation_graph_hourly_forward_ledger import (
    write_v2342,
)


if __name__ == "__main__":
    for name, path in write_v2342().items():
        print(f"{name}: {path}")
