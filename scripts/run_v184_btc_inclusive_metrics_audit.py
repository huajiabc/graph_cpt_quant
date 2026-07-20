from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    write_v184_btc_inclusive_metrics_audit,
)


if __name__ == "__main__":
    for name, path in write_v184_btc_inclusive_metrics_audit().items():
        print(f"{name}: {path}")
