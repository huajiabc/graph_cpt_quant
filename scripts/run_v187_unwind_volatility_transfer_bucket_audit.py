from pressure_graph.reports.v187_unwind_volatility_transfer_bucket_audit import (
    write_v187_audit,
)


if __name__ == "__main__":
    for name, path in write_v187_audit().items():
        print(f"{name}: {path}")
