from pressure_graph.reports.v211_funding_settlement_rebound_audit import (
    write_v211_audit,
)


if __name__ == "__main__":
    for name, path in write_v211_audit().items():
        print(f"{name}: {path}")
