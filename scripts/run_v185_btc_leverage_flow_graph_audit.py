from pressure_graph.reports.v185_btc_leverage_flow_graph_audit import (
    write_v185_audit,
)


if __name__ == "__main__":
    for name, path in write_v185_audit().items():
        print(f"{name}: {path}")
