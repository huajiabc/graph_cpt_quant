from pressure_graph.reports.v174_deribit_skew_receiver_oco import (
    write_v174_deribit_skew_receiver_oco,
)


if __name__ == "__main__":
    for name, path in write_v174_deribit_skew_receiver_oco().items():
        print(f"{name}: {path}")
