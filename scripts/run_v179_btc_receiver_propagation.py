from pressure_graph.reports.v179_btc_receiver_propagation import (
    write_v179_btc_receiver_propagation,
)


if __name__ == "__main__":
    for name, path in write_v179_btc_receiver_propagation().items():
        print(f"{name}: {path}")
