from pressure_graph.reports.v179_btc_receiver_propagation_audit import (
    write_v179_audit,
)


if __name__ == "__main__":
    for name, path in write_v179_audit().items():
        print(f"{name}: {path}")
