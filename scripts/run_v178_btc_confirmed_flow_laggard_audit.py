from pressure_graph.reports.v178_btc_confirmed_flow_laggard_audit import (
    write_v178_audit,
)


if __name__ == "__main__":
    for name, path in write_v178_audit().items():
        print(f"{name}: {path}")
