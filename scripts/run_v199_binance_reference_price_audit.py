from pressure_graph.reports.v199_binance_reference_price_audit import (
    write_v199_reference_price_audit,
)


if __name__ == "__main__":
    for name, path in write_v199_reference_price_audit().items():
        print(f"{name}: {path}")
