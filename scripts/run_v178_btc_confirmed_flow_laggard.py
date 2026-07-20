from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    write_v178_btc_confirmed_flow_laggard,
)


if __name__ == "__main__":
    for name, path in write_v178_btc_confirmed_flow_laggard().items():
        print(f"{name}: {path}")
