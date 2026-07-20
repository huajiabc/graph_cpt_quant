from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    write_v185_btc_leverage_flow_graph,
)


if __name__ == "__main__":
    for name, path in write_v185_btc_leverage_flow_graph().items():
        print(f"{name}: {path}")
