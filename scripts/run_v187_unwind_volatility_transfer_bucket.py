from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    write_v187_unwind_volatility_transfer_bucket,
)


if __name__ == "__main__":
    for name, path in write_v187_unwind_volatility_transfer_bucket().items():
        print(f"{name}: {path}")
