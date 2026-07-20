from pressure_graph.reports.v173_deribit_skew_receiver_bucket import (
    write_v173_deribit_skew_receiver_bucket,
)


if __name__ == "__main__":
    for name, path in write_v173_deribit_skew_receiver_bucket().items():
        print(f"{name}: {path}")
