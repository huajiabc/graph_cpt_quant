from pressure_graph.reports.v190_binance_premium_index_audit import (
    write_v190_premium_audit,
)


if __name__ == "__main__":
    for name, path in write_v190_premium_audit().items():
        print(f"{name}: {path}")
